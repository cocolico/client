from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .action_dit import ActionDiT
from .fastwam_joint import FastWAMJoint
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT


class FastWAMDualActionIDM(FastWAMJoint):
    """FastWAM with future-video, camera-action, and arm-action experts.

    RoboTwin-AV supplies one concatenated action tensor. Its leading arm slice
    supervises the bimanual ``arm_action_expert`` and its trailing camera slice
    supervises the active-camera ``cam_action_expert``. The configured experts
    define the split: joint uses 14+6=20 while native EEF uses 16+7=23. The
    MoT key ``action`` and base attribute ``self.action_expert`` are compatibility
    aliases for the camera branch so existing FastWAMJoint code can be reused.

    During inference, arm coarse denoising reads pre-DiT features from the
    measured observation until ``fusion_sigma``. Video/camera prediction then
    finishes, and the same noisy arm latent resumes with the complete
    predicted-video DiT K/V cache. Training mirrors this information boundary
    with a high-sigma direct-observation loss and a low-sigma future-conditioned
    refinement loss.

    Joint-pose 使用 14D/6D，native EEF 使用 16D/7D；切分由 expert 配置定义，
    不在运行时根据 representation 分支。
    双臂先在高噪声阶段直接读取当前观测的 pre-DiT feature 做粗规划，
    再从同一个 noisy latent 出发，读取预测未来视频的 DiT K/V cache
    完成低噪声精规划。
    """

    VIDEO_EXPERT_KEY = "video"
    # Compatibility key required by FastWAMJoint/MoT. / 复用基类所需的兼容键名。
    CAM_ACTION_EXPERT_KEY = "action"  # Camera alias / 主动相机 expert 的兼容别名
    ARM_ACTION_EXPERT_KEY = "arm_action"  # Dedicated arm key / 双臂 expert 专用键名
    EXPERT_ORDER = (
        VIDEO_EXPERT_KEY,
        CAM_ACTION_EXPERT_KEY,
        ARM_ACTION_EXPERT_KEY,
    )

    def __init__(
        self,
        video_expert,
        cam_action_expert: ActionDiT,
        arm_action_expert: ActionDiT,
        mot: MoT,
        fusion_sigma: float = 0.4,
        arm_action_infer_shift: float | None = None,
        arm_refine_teacher_forcing_prob: float = 1.0,
        arm_refine_teacher_forcing_start_step: int | None = None,
        arm_refine_teacher_forcing_end_step: int | None = None,
        loss_lambda_cam_action: float = 1.0,
        loss_lambda_arm_coarse: float = 1.0,
        loss_lambda_arm_refine: float = 1.0,
        **fastwam_kwargs: Any,
    ) -> None:
        self.cam_action_dim = int(cam_action_expert.action_dim)
        self.arm_action_dim = int(arm_action_expert.action_dim)
        if self.cam_action_dim <= 0 or self.arm_action_dim <= 0:
            raise ValueError("arm and camera action dimensions must be positive")
        self.full_action_dim = self.arm_action_dim + self.cam_action_dim
        configured_proprio_dim = fastwam_kwargs.get("proprio_dim")
        if (
            configured_proprio_dim is not None
            and int(configured_proprio_dim) != self.full_action_dim
        ):
            raise ValueError(
                f"proprio_dim must match full action dim={self.full_action_dim}, "
                f"got {configured_proprio_dim}"
            )
        # FastWAM.__init__ moves ``mot`` to the requested device. Because the
        # arm expert is already registered inside ``mot.mixtures``, it is moved
        # together with video/camera before we add the convenience alias below.
        # arm expert 已注册进 mot.mixtures，因此基类迁移 MoT device/dtype 时会
        # 与 video/camera 一起迁移；这里不需要再次手动移动或重复注册参数。
        super().__init__(
            video_expert=video_expert,
            # FastWAM uses this generic name. In this subclass it is strictly a
            # compatibility alias for the configured camera-action expert.
            # 基类仍使用 action_expert 这个通用名称；在本类中它只代表 camera。
            action_expert=cam_action_expert,
            mot=mot,
            loss_lambda_action=loss_lambda_cam_action,
            **fastwam_kwargs,
        )
        self.cam_action_expert = cam_action_expert
        self.arm_action_expert = arm_action_expert
        self.fusion_sigma = float(fusion_sigma)
        self.arm_action_infer_shift = float(
            self.infer_action_scheduler.shift
            if arm_action_infer_shift is None
            else arm_action_infer_shift
        )
        self.arm_refine_teacher_forcing_prob = float(arm_refine_teacher_forcing_prob)
        self.arm_refine_teacher_forcing_start_step = (
            None
            if arm_refine_teacher_forcing_start_step is None
            else int(arm_refine_teacher_forcing_start_step)
        )
        self.arm_refine_teacher_forcing_end_step = (
            None
            if arm_refine_teacher_forcing_end_step is None
            else int(arm_refine_teacher_forcing_end_step)
        )
        if not 0.0 <= self.arm_refine_teacher_forcing_prob <= 1.0:
            raise ValueError("arm_refine_teacher_forcing_prob must be in [0, 1]")
        schedule_steps = (
            self.arm_refine_teacher_forcing_start_step,
            self.arm_refine_teacher_forcing_end_step,
        )
        if (schedule_steps[0] is None) != (schedule_steps[1] is None):
            raise ValueError(
                "arm_refine_teacher_forcing_start_step and "
                "arm_refine_teacher_forcing_end_step must be set together"
            )
        if schedule_steps[0] is not None:
            if schedule_steps[0] < 0 or schedule_steps[1] <= schedule_steps[0]:
                raise ValueError(
                    "teacher-forcing schedule requires 0 <= start_step < end_step"
                )
        self.loss_lambda_cam_action = float(loss_lambda_cam_action)
        self.loss_lambda_arm_coarse = float(loss_lambda_arm_coarse)
        self.loss_lambda_arm_refine = float(loss_lambda_arm_refine)

    def _arm_refine_teacher_forcing_probability(
        self,
        optimizer_step: int | None,
    ) -> float:
        """Return the GT-future probability for the current optimizer step."""
        start_step = self.arm_refine_teacher_forcing_start_step
        end_step = self.arm_refine_teacher_forcing_end_step
        if start_step is None:
            return self.arm_refine_teacher_forcing_prob
        if optimizer_step is None:
            raise ValueError(
                "optimizer_step is required when the teacher-forcing schedule is enabled"
            )

        optimizer_step = int(optimizer_step)
        if optimizer_step <= start_step:
            return self.arm_refine_teacher_forcing_prob
        if optimizer_step >= end_step:
            return 0.0
        progress = (optimizer_step - start_step) / (end_step - start_step)
        return self.arm_refine_teacher_forcing_prob * (1.0 - progress)

    def split_robotwin_action(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split concatenated RoboTwin actions into named arm/camera views.

        RoboTwin-AV stores the two arms first and the active-camera action last.
        Keeping the split at the model boundary lets each dataset retain its
        native schema and prevents task-specific slicing in shared data loaders.
        """
        if action.shape[-1] != self.full_action_dim:
            raise ValueError(f"action last dim must be {self.full_action_dim}")

        # These are views, not copies: normalization and dtype/device remain
        # exactly those supplied by the shared dataset pipeline.
        # 两个切片都是 view 而不是 copy，因此保持数据管线给出的归一化、dtype 和 device。
        return {
            "arm_action": action[..., : self.arm_action_dim],
            "cam_action": action[..., self.arm_action_dim :],
        }

    def _build_arm_inference_schedule(
        self,
        num_inference_steps: int,
        fusion_sigma: float,
        sigma_shift: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Build an arm schedule that lands exactly on ``fusion_sigma``.

        Returns model timesteps, Euler sigma deltas, and the number of coarse
        steps. Inserting the boundary avoids mapping a requested sigma to the
        nearest loop index when scheduler shifting makes those values differ.

        返回模型 timestep、Euler sigma 增量和 coarse step 数。显式插入边界可
        避免 scheduler shift 后把目标 sigma 错当成“最接近的循环下标”。
        """
        fusion_sigma = float(fusion_sigma)
        if not 0.0 <= fusion_sigma <= 1.0:
            raise ValueError("fusion_sigma must be in [0, 1]")

        timesteps, _ = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=torch.float32,
            # ``sigma_shift`` remains the explicit legacy override for every
            # branch. Otherwise only arm uses its configured inference grid,
            # leaving the camera scheduler unchanged.
            # 显式 sigma_shift 保持全分支覆盖语义；未指定时仅 arm 使用自己的
            # inference grid，camera scheduler 不受影响。
            shift_override=(
                self.arm_action_infer_shift if sigma_shift is None else sigma_shift
            ),
        )
        num_train_timesteps = self.infer_action_scheduler.num_train_timesteps
        native_sigmas = torch.cat([
            timesteps / num_train_timesteps,
            torch.zeros(1, device=self.device),
        ]).clamp_(0.0, 1.0)
        fusion = native_sigmas.new_tensor(fusion_sigma)
        # ``2/3`` maps the middle 4-step node to 0.4 mathematically, but fp32
        # may store it as 0.40000004. Snap a numerically equal native node to
        # the exact boundary before ``unique``; otherwise a meaningless fifth
        # step can appear between these two values.
        # 数学上 shift=2/3 会把四步 schedule 的中间节点映射到 0.4，但 fp32
        # 可能得到 0.40000004。先吸附到精确边界再去重，避免多出一个伪第五步。
        native_sigmas = torch.where(
            torch.isclose(native_sigmas, fusion, rtol=1e-5, atol=1e-6),
            fusion,
            native_sigmas,
        )
        sigmas = torch.unique(torch.cat([native_sigmas, fusion[None]])).sort(
            descending=True
        ).values
        timesteps = sigmas[:-1] * num_train_timesteps
        deltas = sigmas[1:] - sigmas[:-1]
        coarse_steps = int((sigmas[:-1] > fusion_sigma).sum().item())
        return timesteps, deltas, coarse_steps

    def _prepare_arm_observation_features(
        self,
        observation_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Convert the measured VAE latent into pre-DiT observation tokens.

        This method intentionally stops after the pretrained Video-DiT patch
        embedding. Running ``mot.prefill_video_cache_tensor`` here would change
        the intended coarse planner back into an observation-only Video-DiT K/V
        planner.

        这里只复用 Video DiT 的 patch embedding，把实测 VAE latent 转成
        observation feature token；严禁在 coarse 路径调用 Video-DiT block
        或构建 layer-wise K/V cache。
        """
        observation_grid = self.video_expert.patchify(observation_latents)
        return observation_grid.flatten(2).transpose(1, 2).contiguous()

    def _denoise_arm_action_with_observation_features(
        self,
        arm_action_latents: torch.Tensor,
        observation_features: torch.Tensor,
        timesteps: torch.Tensor,
        deltas: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise arm latents while reading direct observation features only."""
        for step_t_action, step_delta_action in zip(timesteps, deltas):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=arm_action_latents.dtype,
                device=self.device,
            )
            pred_arm_action = self.arm_action_expert(
                action_tokens=arm_action_latents,
                timestep=timestep_action,
                observation_features=observation_features,
                context=context,
                context_mask=context_mask,
            )
            arm_action_latents = self.infer_action_scheduler.step(
                pred_arm_action,
                step_delta_action,
                arm_action_latents,
            )
        return arm_action_latents

    def _denoise_arm_action_with_video_cache(
        self,
        arm_action_latents: torch.Tensor,
        video_latents: torch.Tensor,
        timesteps: torch.Tensor,
        deltas: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_flag: bool,
    ) -> torch.Tensor:
        """Refine arm latents against a fixed predicted-video K/V cache.

        ``video_latents`` contains the measured first frame plus predicted
        future frames. Reusing one cache throughout refinement keeps the
        low-sigma conditioning evidence fixed.

        refine 阶段传入实测首帧加预测未来，并固定使用一份 layer-wise Video
        K/V cache；coarse 阶段不会调用本方法。
        """
        timestep_video = torch.zeros(1, dtype=video_latents.dtype, device=self.device)
        (
            video_tokens,
            _t_video,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            _f,
            _h,
            _w,
            video_tokens_per_frame,
        ) = self.video_expert.prepare(
            x=video_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_tokens.shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=arm_action_latents.shape[1],
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_tokens.device,
        )
        video_cache_k, video_cache_v = self.mot.prefill_video_cache_tensor(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        action_attention_mask = attention_mask[video_seq_len:, :]

        for step_t_action, step_delta_action in zip(timesteps, deltas):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=arm_action_latents.dtype, device=self.device
            )
            (
                action_tokens,
                _t_action,
                action_t_mod,
                action_context,
                action_context_mask,
                action_freqs,
            ) = self.arm_action_expert.prepare(
                action_tokens=arm_action_latents,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )
            action_tokens = self.mot.forward_action_with_video_cache_tensor(
                action_tokens=action_tokens,
                action_freqs=action_freqs,
                action_t_mod=action_t_mod,
                action_context=action_context,
                action_context_mask=action_context_mask,
                video_cache_k=video_cache_k,
                video_cache_v=video_cache_v,
                action_attention_mask=action_attention_mask,
                action_expert_name=self.ARM_ACTION_EXPERT_KEY,
            )
            pred_arm_action = self.arm_action_expert.post(action_tokens)
            arm_action_latents = self.infer_action_scheduler.step(
                pred_arm_action,
                step_delta_action,
                arm_action_latents,
            )
        return arm_action_latents

    def build_arm_video_cache(
        self,
        video_latents: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_flag: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
        """Build the clean layer-wise video K/V cache for arm refinement.

        The cache contains the measured first frame plus GT/predicted future
        latents. Coarse planning never calls this method; it uses direct pre-DiT
        observation features instead.

        cache 包含实测首帧与 GT/预测未来 latent，只供 refinement 使用；
        coarse planning 直接读取 pre-DiT observation feature，不调用本方法。
        """
        (
            video_tokens,
            _t_video,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            _f,
            _h,
            _w,
            video_tokens_per_frame,
        ) = self.video_expert.prepare(
            x=video_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_tokens.shape[1],
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_tokens.device,
        )
        video_cache_k, video_cache_v = self.mot.prefill_video_cache_tensor(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
        )
        return video_cache_k, video_cache_v, video_tokens_per_frame

    def _arm_training_denoise_core(
        self,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        video_tokens_per_frame: int,
        noisy_arm_action: torch.Tensor,
        timestep_arm: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict refinement flow from a prepared future-video K/V cache.

        This core is intentionally refinement-only. Coarse planning calls the
        normal ActionDiT path with direct observation features, so a caller that
        reaches this method has explicitly crossed the ``fusion_sigma`` boundary.

        本方法只服务 refinement。coarse planning 通过普通 ActionDiT 直接读取
        observation feature；调用到这里就表示已经越过 ``fusion_sigma`` 边界。
        """
        (
            arm_tokens,
            _t_arm,
            arm_t_mod,
            arm_context,
            arm_context_mask,
            arm_freqs,
        ) = self.arm_action_expert.prepare(
            action_tokens=noisy_arm_action,
            timestep=timestep_arm,
            context=context,
            context_mask=context_mask,
        )

        video_seq_len = int(video_cache_k[0].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=arm_tokens.shape[1],
            video_tokens_per_frame=video_tokens_per_frame,
            device=arm_tokens.device,
        )
        arm_tokens = self.mot.forward_action_with_video_cache_tensor(
            action_tokens=arm_tokens,
            action_freqs=arm_freqs,
            action_t_mod=arm_t_mod,
            action_context=arm_context,
            action_context_mask=arm_context_mask,
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            action_attention_mask=attention_mask[video_seq_len:, :],
            action_expert_name=self.ARM_ACTION_EXPERT_KEY,
        )
        return self.arm_action_expert.post(arm_tokens)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        cam_action_dit_config: dict[str, Any] | None = None,
        arm_action_dit_config: dict[str, Any] | None = None,
        cam_action_dit_pretrained_path: str | None = None,
        arm_action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        shared_action_train_shift: float = 1.0,
        cam_action_infer_shift: float = 1.0,
        arm_action_infer_shift: float | None = None,
        action_num_train_timesteps: int = 1000,
        fusion_sigma: float = 0.4,
        arm_refine_teacher_forcing_prob: float = 1.0,
        arm_refine_teacher_forcing_start_step: int | None = None,
        arm_refine_teacher_forcing_end_step: int | None = None,
        loss_lambda_video: float = 1.0,
        loss_lambda_cam_action: float = 1.0,
        loss_lambda_arm_coarse: float = 1.0,
        loss_lambda_arm_refine: float = 1.0,
        compile_training_denoise: bool = False,
        action_video_alignment_mode: str = "uniform",
    ):
        """Load one video expert and two independent ActionDiT experts.

        两个 ActionDiT 可以复用同一 backbone 文件，但会分别构造与加载；
        camera 与 arm encoder/head 根据各自 action_dim 独立随机初始化。
        """
        if video_dit_config is None:
            raise ValueError("video_dit_config is required")
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError("video_dit_config.action_conditioned must be false")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        video_expert = components.dit
        cam_action_expert = ActionDiT.from_pretrained(
            action_dit_config=cam_action_dit_config,
            action_dit_pretrained_path=cam_action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        arm_action_expert = ActionDiT.from_pretrained(
            action_dit_config=arm_action_dit_config,
            action_dit_pretrained_path=arm_action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        # Mixed attention requires identical layer/head geometry. Action width
        # may differ because it is handled by each expert's private encoder/head.
        # MoT 混合注意力要求层数与 head geometry 一致；action_dim 可独立配置。
        for name, expert in (
            ("cam_action_expert", cam_action_expert),
            ("arm_action_expert", arm_action_expert),
        ):
            if int(expert.num_heads) != int(video_expert.num_heads):
                raise ValueError(f"{name}.num_heads must match video_expert")
            if int(expert.attn_head_dim) != int(video_expert.attn_head_dim):
                raise ValueError(f"{name}.attn_head_dim must match video_expert")
            if len(expert.blocks) != len(video_expert.blocks):
                raise ValueError(f"{name}.num_layers must match video_expert")

        # Keep the legacy 'action' key as the camera alias; arm_action is a new,
        # independent route. / action 保持为 camera 兼容别名，arm_action 使用独立路由。
        mot = MoT(
            mixtures={
                cls.VIDEO_EXPERT_KEY: video_expert,
                cls.CAM_ACTION_EXPERT_KEY: cam_action_expert,
                cls.ARM_ACTION_EXPERT_KEY: arm_action_expert,
            },
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            cam_action_expert=cam_action_expert,
            arm_action_expert=arm_action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            # Camera and arm sample training sigmas from this same scheduler.
            # camera 与 arm 的训练 sigma 都从同一个 scheduler 采样。
            action_train_shift=shared_action_train_shift,
            # FastWAM names its single action scheduler generically. In this
            # subclass that scheduler belongs to the camera branch; arm uses
            # ``arm_action_infer_shift`` when building its own sigma grid.
            # 基类的 action scheduler 在本模型中对应 camera；arm 只复用其实现，
            # 并用独立的 ``arm_action_infer_shift`` 构造推理节点。
            action_infer_shift=cam_action_infer_shift,
            arm_action_infer_shift=arm_action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            fusion_sigma=fusion_sigma,
            arm_refine_teacher_forcing_prob=arm_refine_teacher_forcing_prob,
            arm_refine_teacher_forcing_start_step=arm_refine_teacher_forcing_start_step,
            arm_refine_teacher_forcing_end_step=arm_refine_teacher_forcing_end_step,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_cam_action=loss_lambda_cam_action,
            loss_lambda_arm_coarse=loss_lambda_arm_coarse,
            loss_lambda_arm_refine=loss_lambda_arm_refine,
            compile_training_denoise=compile_training_denoise,
            action_video_alignment_mode=action_video_alignment_mode,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "cam_action_dit_backbone": (
                "SKIPPED_PRETRAIN"
                if skip_dit_load_from_pretrain
                else cam_action_dit_pretrained_path
            ),
            "arm_action_dit_backbone": (
                "SKIPPED_PRETRAIN"
                if skip_dit_load_from_pretrain
                else arm_action_dit_pretrained_path
            ),
        }
        return model

    def training_loss(
        self,
        sample,  # dataloader batch dict
        tiled: bool = False,  # 控制 vae 是否切片后再编码，缓解显存压力
        optimizer_step: int | None = None,
    ):
        """Compute video, camera, arm-coarse, and arm-refinement objectives.

        Coarse arm sigma is sampled from ``[fusion_sigma, 1]`` and reads direct
        pre-DiT observation features. Refinement sigma is sampled from
        ``[0, fusion_sigma]`` and reads GT or predicted future Video-DiT K/V
        according to the scheduled teacher-forcing probability. Returned loss
        metrics are already weighted.

        arm coarse 从 ``[fusion_sigma, 1]`` 采样，直接读取 pre-DiT observation
        feature；arm refine 从 ``[0, fusion_sigma]`` 采样，并按 teacher-forcing
        调度读取 GT 或预测未来的 Video-DiT K/V。预测分支保留完整梯度。
        """
        # 处理 dataloader batch，构建模型:
        # 1. VAE 编码视频 latents
        # 2. 编码文本/任务条件
        # 其他包括处理 proprio, mask, padding 等 并移动到模型 device/dtype
        inputs = self.build_inputs(sample, tiled=tiled)

        # 提取各个字段
        input_latents = inputs["input_latents"] # [B,C,F,H,W] / VAE 编码后的视频
        context = inputs["context"] # [B,L,D] / 文本/任务条件
        context_mask = inputs["context_mask"] # [B,L] / 有效文本 token
        action_is_pad = inputs["action_is_pad"] # [B,T] / action padding mask
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"] # video expert 是否把 VAE embedding 融入 latent token
        batch_size = input_latents.shape[0]
        actions = self.split_robotwin_action(inputs["action"])
        cam_action = actions["cam_action"]
        arm_action = actions["arm_action"]
        action_valid_mask = inputs.get("action_valid_mask")
        if action_valid_mask is None:
            arm_valid_mask = None
            cam_valid_mask = None
        else:
            action_masks = self.split_robotwin_action(action_valid_mask)
            arm_valid_mask = action_masks["arm_action"]
            cam_valid_mask = action_masks["cam_action"]

        # Video/camera keep FastWAMJoint's directed information flow: video reads
        # observation/context, while the camera expert may attend to video.
        # 保持 FastWAMJoint 的有向信息流：video 读取观测与任务上下文，camera
        # expert 可以关注 video，但 video 不读取 camera action。

        # 构建 video flow-matching 的噪声样本
        noise_video = torch.randn_like(input_latents)
        # 采样一个 timestep，添加噪声，计算目标
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        noisy_video = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        # 如果任务使用首帧 conditioning，就把视频 latent 的第一个时间步恢复成真实观测
        if inputs["first_frame_latents"] is not None:
            noisy_video[:, :, 0:1] = inputs["first_frame_latents"]

        noise_cam_action = torch.randn_like(cam_action)
        timestep_cam = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=cam_action.dtype,
        )
        noisy_cam_action = self.train_action_scheduler.add_noise(
            cam_action, noise_cam_action, timestep_cam
        )
        target_cam_action = self.train_action_scheduler.training_target(
            cam_action, noise_cam_action, timestep_cam
        )

        # 构造 video/cam_action 联合 attention
        # MoT attention mask，确保 video tokens 只关注自身，action tokens 可以关注 video
        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        latent_t, latent_h, latent_w = noisy_video.shape[-3:]
        # 计算每个 latent frame 被切成多少个 token：
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)

        # 构建 attention mask，确保 video tokens 只关注自身，action tokens 可以关注 video
        # video token --> video token
        # cam_action token  --> video token + cam_action token
        # video token -!> cam_action token
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=noisy_cam_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=noisy_video.device,
        )
        # 第一次主要 forward
        pred_video, pred_cam_action = self._joint_denoise_core(
            latents_video=noisy_video,
            latents_action=noisy_cam_action,
            timestep_video=timestep_video,
            timestep_action=timestep_cam,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            action_condition=None,
        )

        observation_latents = input_latents[:, :, 0:1]  # Measured frame / 实测首帧
        # Coarse conditioning stops after Video-DiT patch embedding. These
        # features are sent to arm cross-attention through the configured
        # adapter; no Video-DiT block or K/V cache is involved.
        # coarse 条件只经过 Video-DiT patch embedding，再由配置化 adapter 送入
        # arm cross-attention；这里不会执行 Video-DiT block 或构建 K/V cache。
        observation_features = self._prepare_arm_observation_features(
            observation_latents
        )
        # When teacher forcing is disabled for a sample, reconstruct the video
        # branch's own one-step x0 estimate: x_sigma = x0 + sigma * v.
        # Keep this path differentiable so arm refinement can shape the future-video
        # prediction it consumes. / 保留完整梯度，使 arm refinement 能优化其读取的
        # future-video prediction。
        sigma_video = timestep_video.float() / float(
            self.train_video_scheduler.num_train_timesteps
        ) # 把离散 timestep 转换为真实 σ = t_v / N
        sigma_video = sigma_video.view(batch_size, 1, 1, 1, 1)
        predicted_video = (
            noisy_video.float() - sigma_video * pred_video.float()
        ).to(input_latents.dtype)
        predicted_video[:, :, 0:1] = observation_latents

        teacher_forcing_prob = self._arm_refine_teacher_forcing_probability(
            optimizer_step
        )
        # teacher-forcing 随机选择
        use_teacher_forcing = (
            torch.rand((batch_size, 1, 1, 1, 1), device=self.device)
            < teacher_forcing_prob
        )
        # Select one condition per sample. / 每个 sample 独立选择 GT 或 predicted 条件。
        refine_video = torch.where(
            use_teacher_forcing,
            input_latents,
            predicted_video,
        )

        # Both arm phases train the same arm expert with the same sampled noise.
        # Only sigma range and video evidence differ, isolating the intended
        # coarse-to-refine behavior instead of learning two unrelated samples.
        # 两个阶段训练同一个 arm expert，并共享同一份 sampled noise；仅 sigma
        # 区间与可见视频证据不同，保证它们是连续粗到精规划而非两个独立样本。
        noise_arm_action = torch.randn_like(arm_action)
        base_timestep_coarse = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=arm_action.dtype,
        )
        base_timestep_refine = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=arm_action.dtype,
        )
        num_train_timesteps = float(self.train_action_scheduler.num_train_timesteps)
        # Map base sigma u to the two target intervals. / 将基础 sigma u 映射到两段区间。
        coarse_sigma = self.fusion_sigma + (1.0 - self.fusion_sigma) * (
            base_timestep_coarse / num_train_timesteps
        )
        refine_sigma = self.fusion_sigma * (
            base_timestep_refine / num_train_timesteps
        )
        timestep_arm_coarse = coarse_sigma * num_train_timesteps
        timestep_arm_refine = refine_sigma * num_train_timesteps

        # 两者独立采样，但使用相同的 action 和 noise。
        target_arm_action = self.train_action_scheduler.training_target(
            arm_action, noise_arm_action, timestep_arm_coarse
        )
        noisy_arm_action_coarse = self.train_action_scheduler.add_noise(
            arm_action, noise_arm_action, timestep_arm_coarse
        )
        noisy_arm_action_refine = self.train_action_scheduler.add_noise(
            arm_action, noise_arm_action, timestep_arm_refine
        )
        # Coarse forward / 第二次 forward，粗规划阶段
        pred_arm_action_coarse = self.arm_action_expert(
            action_tokens=noisy_arm_action_coarse,
            timestep=timestep_arm_coarse,
            observation_features=observation_features,
            context=context,
            context_mask=context_mask,
        )

        # Refinement cache represents clean future conditioning evidence.
        # refinement cache 表示干净的未来条件证据。
        timestep_video_cond = torch.zeros_like(timestep_video) # 告诉 video expert： 当前 cache 中的 video latent 是干净条件，不需要再当作 noisy video 处理

        # Only refinement builds a Video-DiT cache. Coarse uses direct
        # observation features above, so no first-frame K/V slice is created.
        # 只有 refinement 构建 Video-DiT cache；coarse 已直接使用 observation
        # feature，因此不再截取 first-frame K/V。
        video_cache_k, video_cache_v, video_tokens_per_frame = self.build_arm_video_cache(
            video_latents=refine_video,
            timestep_video=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            fuse_flag=fuse_flag,
        )

        # Refinement forward / 第三次 forward，精规划阶段
        pred_arm_action_refine = self._arm_training_denoise_core(
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            video_tokens_per_frame=video_tokens_per_frame,
            noisy_arm_action=noisy_arm_action_refine,
            timestep_arm=timestep_arm_refine,
            context=context,
            context_mask=context_mask,
        )

        # 计算 video loss
        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        # 统一计算三个 action loss
        action_losses = {}
        # Apply padding mask and scheduler weighting uniformly to all actions.
        # 三项 action loss 统一使用 padding mask 和 scheduler timestep 权重。
        for name, prediction, target, timestep, valid_dimensions in (
            ("cam_action", pred_cam_action, target_cam_action, timestep_cam, cam_valid_mask),
            ("arm_action_coarse", pred_arm_action_coarse, target_arm_action, timestep_arm_coarse, arm_valid_mask),
            ("arm_action_refine", pred_arm_action_refine, target_arm_action, timestep_arm_refine, arm_valid_mask),
        ):
            loss_dimensions = F.mse_loss(
                prediction.float(), target.float(), reduction="none"
            )
            if valid_dimensions is None:
                loss_token = loss_dimensions.mean(dim=2)
                token_valid = torch.ones_like(loss_token, dtype=torch.bool)
            else:
                valid_float = valid_dimensions.to(loss_dimensions.dtype)
                loss_token = (loss_dimensions * valid_float).sum(dim=2) / valid_float.sum(
                    dim=2
                ).clamp(min=1)
                token_valid = valid_dimensions.any(dim=2)
            if action_is_pad is not None:
                token_valid = token_valid & (~action_is_pad)
            if action_is_pad is not None or valid_dimensions is not None:
                valid = token_valid.to(loss_token.dtype)
                loss_per_sample = (loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            else:
                loss_per_sample = loss_token.mean(dim=1)
            weight = self.train_action_scheduler.training_weight(timestep).to(
                loss_per_sample.device, dtype=loss_per_sample.dtype
            )
            action_losses[name] = (loss_per_sample * weight).mean()

        # Keep explicit knobs for later balancing. / 保留独立旋钮，便于后续平衡各分支。
        weighted_losses = {
            "loss_video": self.loss_lambda_video * loss_video,
            "loss_cam_action": self.loss_lambda_cam_action * action_losses["cam_action"],
            "loss_arm_action_coarse": self.loss_lambda_arm_coarse * action_losses["arm_action_coarse"],
            "loss_arm_action_refine": self.loss_lambda_arm_refine * action_losses["arm_action_refine"],
        }

        # Compute total loss and log metrics. / 计算总 loss 并记录日志指标。
        loss_total = sum(weighted_losses.values())
        loss_dict = {
            name: float(loss.detach().item())
            for name, loss in weighted_losses.items()
        }
        # Log actual sampled behavior, not only configured probabilities/ranges.
        # 记录本 batch 的真实采样比例与 sigma，而不只记录静态配置值。
        loss_dict["arm_refine_teacher_forcing_ratio"] = float(
            use_teacher_forcing.float().mean().item()
        )
        loss_dict["arm_refine_teacher_forcing_prob"] = teacher_forcing_prob
        loss_dict["video_sigma"] = float(sigma_video.mean().item())
        loss_dict["arm_coarse_sigma"] = float(coarse_sigma.mean().item())
        loss_dict["arm_refine_sigma"] = float(refine_sigma.mean().item())
        return loss_total, loss_dict

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        cam_action_horizon: int,
        arm_action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        fusion_sigma: float = 0.4,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
        arm_refine_use_future_video: bool = True,
        parallel_coarse_video: bool = False,
    ) -> dict[str, Any]:
        """Predict future video plus staged camera and arm actions.

        ``arm_action_latent_at_fusion`` is returned for diagnostics only: unless
        fusion sigma is zero, it remains noisy and is not an executable action.

        返回 future video、camera action 与 arm action。中间的
        arm_action_latent_at_fusion 仅用于诊断；fusion_sigma 非零时它仍含噪，
        不能作为可执行动作。
        """
        del negative_prompt, text_cfg_scale, compile_action_infer
        self.eval()

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"input_image must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height,
            width,
            num_video_frames,
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(f"input_image H/W must be multiples of 16, got ({height},{width})")
        if checked_t != num_video_frames:
            raise ValueError(f"num_video_frames must satisfy T % 4 == 1, got {num_video_frames}")

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("proprio_dim is required when proprio is provided")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim != 2 or proprio.shape[0] != 1:
                raise ValueError(f"proprio must be [D] or [1,D], got {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"proprio last dim must be {self.proprio_dim}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (
            (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        )
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        cam_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        arm_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        video_latents = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        cam_action_latents = torch.randn(
            (1, cam_action_horizon, self.cam_action_dim),
            generator=cam_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        arm_action_latents = torch.randn(
            (1, arm_action_horizon, self.arm_action_dim),
            generator=arm_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        )
        video_latents[:, :, 0:1] = first_frame_latents.clone()
        # Encode the measured frame once before optional CUDA-stream overlap.
        # Coarse arm planning reads these pre-DiT features directly, while the
        # video/camera stream remains free to predict the future independently.
        # 在可选 CUDA 并行之前只编码一次实测帧；arm coarse 直接读取 pre-DiT
        # feature，video/camera stream 则独立完成未来预测。
        observation_features = self._prepare_arm_observation_features(
            first_frame_latents
        )
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("prompt and context/context_mask are mutually exclusive")
        if not use_prompt and not use_context:
            raise ValueError("prompt or context/context_mask must be provided")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("context and context_mask must be provided together")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        arm_timesteps, arm_deltas, coarse_steps = self._build_arm_inference_schedule(
            num_inference_steps=num_inference_steps,
            fusion_sigma=fusion_sigma,
            sigma_shift=sigma_shift,
        )

        # Arm coarse and video/camera only share read-only observation/context.
        # Two CUDA streams can therefore overlap them without changing either
        # denoising trajectory. CPU and the default flag retain serial execution.
        # arm coarse 与 video/camera 只共享只读的观测和 context，因此 CUDA 下可
        # 放到两个 stream 并行；CPU 或默认关闭开关时仍保持原来的串行路径。
        run_parallel = bool(
            parallel_coarse_video
            and coarse_steps
            and self.device.type == "cuda"
        )
        current_stream = torch.cuda.current_stream(self.device) if run_parallel else None
        arm_stream = torch.cuda.Stream(device=self.device) if run_parallel else None
        video_stream = torch.cuda.Stream(device=self.device) if run_parallel else None
        if run_parallel:
            arm_stream.wait_stream(current_stream)
            video_stream.wait_stream(current_stream)

        arm_stream_context = (
            torch.cuda.stream(arm_stream) if run_parallel else nullcontext()
        )
        with arm_stream_context:
            # Phase 1 / 粗规划：arm 直接 cross-attend 当前观测 feature；不运行
            # Video-DiT block，也不读取任何 Video K/V 或未来信息。
            if coarse_steps:
                arm_action_latents = self._denoise_arm_action_with_observation_features(
                    arm_action_latents=arm_action_latents,
                    observation_features=observation_features,
                    timesteps=arm_timesteps[:coarse_steps],
                    deltas=arm_deltas[:coarse_steps],
                    context=context,
                    context_mask=context_mask,
                )
            # Still noisy when fusion_sigma > 0. / fusion_sigma>0 时不是可执行动作。
            arm_action_latent_at_fusion = arm_action_latents.detach().clone()

        # Video and camera follow the original FastWAMJoint denoising path.
        # Video/camera 沿用原始 FastWAMJoint 联合去噪路径，不引入第二套实现。
        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=cam_action_horizon,
            video_tokens_per_frame=tokens_per_frame,
            device=self.device,
        )
        video_timesteps, video_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=video_latents.dtype,
            shift_override=sigma_shift,
        )
        cam_timesteps, cam_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=cam_action_latents.dtype,
            shift_override=sigma_shift,
        )
        video_stream_context = (
            torch.cuda.stream(video_stream) if run_parallel else nullcontext()
        )
        with video_stream_context:
            for step_t_video, step_delta_video, step_t_cam, step_delta_cam in zip(
                video_timesteps,
                video_deltas,
                cam_timesteps,
                cam_deltas,
            ):
                timestep_video = step_t_video.unsqueeze(0).to(
                    dtype=video_latents.dtype,
                    device=self.device,
                )
                timestep_cam = step_t_cam.unsqueeze(0).to(
                    dtype=cam_action_latents.dtype,
                    device=self.device,
                )
                pred_video, pred_cam_action = self._joint_denoise_core(
                    latents_video=video_latents,
                    latents_action=cam_action_latents,
                    timestep_video=timestep_video,
                    timestep_action=timestep_cam,
                    context=context,
                    context_mask=context_mask,
                    attention_mask=attention_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    action_condition=None,
                )
                video_latents = self.infer_video_scheduler.step(
                    pred_video,
                    step_delta_video,
                    video_latents,
                )
                cam_action_latents = self.infer_action_scheduler.step(
                    pred_cam_action,
                    step_delta_cam,
                    cam_action_latents,
                )
                # Never update measured frame. / 首帧是真实观测，任何去噪 step 都不能修改。
                video_latents[:, :, 0:1] = first_frame_latents.clone()

        if run_parallel:
            # Fusion barrier: the current stream waits only for the two producers;
            # no device-wide synchronize is needed before arm refinement.
            # Fusion barrier 只等待两个生产 stream，不做全 GPU 同步。
            current_stream.wait_stream(arm_stream)
            current_stream.wait_stream(video_stream)

        # Phase 2 / 精规划：默认切换到 [当前观测 + 预测未来] 的完整 Video-DiT
        # K/V cache。关闭 future-video 消融时，则继续使用同一份 direct
        # observation feature，绝不退回 observation-only Video-DiT K/V。
        # The arm latent resumes from exactly fusion_sigma, so this is inverse
        # dynamics refinement rather than a second independent action sample.
        # 从 fusion_sigma 处暂停的同一个 arm latent 继续，因此这是 future-video
        # 引导的 inverse-dynamics refinement，而不是重新采样另一条 arm action。
        if coarse_steps < arm_timesteps.numel():
            if arm_refine_use_future_video:
                arm_action_latents = self._denoise_arm_action_with_video_cache(
                    arm_action_latents=arm_action_latents,
                    video_latents=video_latents,
                    timesteps=arm_timesteps[coarse_steps:],
                    deltas=arm_deltas[coarse_steps:],
                    context=context,
                    context_mask=context_mask,
                    fuse_flag=fuse_flag,
                )
            else:
                # Observation-only ablation keeps the direct feature path for
                # the whole trajectory; predicted video/camera outputs are
                # still produced but cannot condition the arm.
                # observation-only 消融让整条 arm 轨迹始终读取 direct feature；
                # video/camera 仍正常预测，但其输出不能进入 arm。
                arm_action_latents = self._denoise_arm_action_with_observation_features(
                    arm_action_latents=arm_action_latents,
                    observation_features=observation_features,
                    timesteps=arm_timesteps[coarse_steps:],
                    deltas=arm_deltas[coarse_steps:],
                    context=context,
                    context_mask=context_mask,
                )

        return {
            "cam_action": cam_action_latents[0].detach().to(
                device="cpu",
                dtype=torch.float32,
            ),
            "arm_action": arm_action_latents[0].detach().to(
                device="cpu",
                dtype=torch.float32,
            ),
            "arm_action_latent_at_fusion": arm_action_latent_at_fusion[0].detach().to(
                device="cpu",
                dtype=torch.float32,
            ),
            "video_latents": video_latents.detach(),
            "future_video_latents": video_latents[:, :, 1:].detach(),
            "fusion_sigma": float(fusion_sigma),
            "arm_refine_use_future_video": bool(arm_refine_use_future_video),
            "parallel_coarse_video": run_parallel,
            "arm_coarse_steps": coarse_steps,
            "arm_refine_steps": int(arm_timesteps.numel()) - coarse_steps,
        }

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        cam_action_horizon: int,
        arm_action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        fusion_sigma: float = 0.4,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
        arm_refine_use_future_video: bool = True,
        parallel_coarse_video: bool = False,
    ) -> dict[str, Any]:
        """Run staged inference and additionally decode the predicted video."""
        output = self.infer_action(
            prompt=prompt,
            input_image=input_image,
            cam_action_horizon=cam_action_horizon,
            arm_action_horizon=arm_action_horizon,
            num_video_frames=num_video_frames,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            fusion_sigma=fusion_sigma,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            compile_action_infer=compile_action_infer,
            arm_refine_use_future_video=arm_refine_use_future_video,
            parallel_coarse_video=parallel_coarse_video,
        )
        return {
            **output,
            "video": self._decode_latents(output["video_latents"], tiled=tiled),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        cam_action_horizon: int,
        arm_action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        fusion_sigma: float = 0.4,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
        arm_refine_use_future_video: bool = True,
        parallel_coarse_video: bool = False,
    ) -> dict[str, Any]:
        """Public decoded-video alias with explicit dual-action horizons."""
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            cam_action_horizon=cam_action_horizon,
            arm_action_horizon=arm_action_horizon,
            num_video_frames=num_frames,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            fusion_sigma=fusion_sigma,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            compile_action_infer=compile_action_infer,
            arm_refine_use_future_video=arm_refine_use_future_video,
            parallel_coarse_video=parallel_coarse_video,
        )

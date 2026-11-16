#!/usr/bin/env python3
"""独立控制一只 Sharpa Wave SE 手；不读取或执行旧 19 维模型动作。"""
from __future__ import annotations

import argparse
import json
import math
import time

from litchi_hardware import sharpa
from litchi_hardware.core.capabilities import HandControlMode
from litchi_hardware.core.types import HandSide

SCHEMA = sharpa.SHARPA_WAVE_JOINT_SCHEMA
MAX_COMMAND_DELTA_RAD = 0.1


def validated_state(hand):
    state = hand.read_joint_state()
    if state.schema is None or state.schema.id != SCHEMA.id or tuple(state.names) != SCHEMA.names:
        raise RuntimeError('Sharpa 关节 schema 或顺序与 22 维 Wave SE 不一致')
    positions = tuple(float(v) for v in state.position)
    if len(positions) != SCHEMA.size or not all(math.isfinite(v) for v in positions):
        raise RuntimeError('Sharpa 关节反馈缺失或含非有限值')
    return state, positions


def run_one(args, side: str) -> dict:
    hand = sharpa.create_hand(
        sharpa.SharpaConfig(
            name=f'agibot-{side}',
            side=HandSide(side),
            serial=args.serial,
            enable_tactile=False,
            hand_variant='se' if args.fake else None,
        ),
        use_fake_hardware=args.fake,
    )
    try:
        # 真机 connect() 会使能手部并切换到 POSITION；read 也会发生此操作。
        hand.connect()
        if hand.control_mode is not HandControlMode.POSITION:
            raise RuntimeError(f'{side} 手未处于 POSITION 模式')
        state, current = validated_state(hand)
        result = {
            'side': side, 'serial': hand.info.serial,
            'schema': SCHEMA.id, 'timestamp_s': state.timestamp_s,
            'position': dict(zip(SCHEMA.names, current, strict=True)),
        }
        if args.mode == 'read':
            return result
        joint_index = SCHEMA.names.index(args.joint)
        target_value = current[joint_index] + args.delta_rad
        if not math.isfinite(target_value):
            raise ValueError('目标角度必须为有限值')
        lower, upper = SCHEMA.limits[joint_index]
        if not lower <= target_value <= upper:
            raise ValueError(f'{args.joint} 目标 {target_value:.4f} rad 超出 [{lower:.4f}, {upper:.4f}]')
        target = list(current)
        target[joint_index] = target_value
        hand.command_joint_position(tuple(target))
        deadline = time.monotonic() + args.timeout
        # SDK 异步发送；给硬件一帧时间处理命令，避免旧反馈被当成到位。
        time.sleep(0.05)
        while True:
            state, observed = validated_state(hand)
            error = abs(observed[joint_index] - target_value)
            if error <= args.tolerance:
                result.update(joint=args.joint, start_rad=current[joint_index],
                              target_rad=target_value, observed_rad=observed[joint_index],
                              error_rad=error, reached=True)
                return result
            if time.monotonic() >= deadline:
                raise RuntimeError(f'{side}/{args.joint} 未在 {args.timeout}s 内到位；目标 {target_value:.4f}，实测 {observed[joint_index]:.4f} rad')
            time.sleep(0.05)
    finally:
        if hand.is_connected:
            try:
                hand.stop()
            finally:
                hand.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('schema', 'read', 'move'))
    parser.add_argument('--side', choices=('left', 'right', 'both'), default='both')
    parser.add_argument('--serial', help='指定单只手的序列号；--side both 时不可使用')
    parser.add_argument('--joint', choices=SCHEMA.names, help='move 时要移动的单个关节')
    parser.add_argument('--delta-rad', type=float, help='相对当前反馈移动的弧度，绝对值不超过 0.1')
    parser.add_argument('--timeout', type=float, default=3.0)
    parser.add_argument('--tolerance', type=float, default=0.005)
    parser.add_argument('--fake', action='store_true', help='SDK 模拟手，不连接真机')
    parser.add_argument('--execute', action='store_true', help='允许真机位置命令')
    args = parser.parse_args()
    if args.mode == 'schema':
        print(json.dumps({'schema': SCHEMA.id, 'unit': SCHEMA.unit,
                          'joints': [{'name': name, 'limits_rad': limit}
                                     for name, limit in zip(SCHEMA.names, SCHEMA.limits, strict=True)]},
                         ensure_ascii=False))
        return
    if args.serial and args.side == 'both':
        parser.error('--serial 需要指定 --side left 或 right')
    if args.mode == 'move':
        if args.side == 'both':
            parser.error('move 必须明确指定 --side left 或 right')
        if args.joint is None or args.delta_rad is None:
            parser.error('move 需要 --joint 和 --delta-rad')
        if not math.isfinite(args.delta_rad) or abs(args.delta_rad) > MAX_COMMAND_DELTA_RAD or args.delta_rad == 0:
            parser.error('--delta-rad 必须为非零有限值，且绝对值不超过 0.1 rad')
        if abs(args.delta_rad) <= 2 * args.tolerance:
            parser.error('位移必须大于到位容差的两倍，避免旧反馈被误判为成功')
        if not args.fake and not args.execute:
            parser.error('真机位置控制需要 --execute')
    if args.timeout <= 0 or not 0 < args.tolerance <= 0.1:
        parser.error('timeout/tolerance 范围无效')
    sides = ('left', 'right') if args.side == 'both' else (args.side,)
    for side in sides:
        print(json.dumps(run_one(args, side), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()

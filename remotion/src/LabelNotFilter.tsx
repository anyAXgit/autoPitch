import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, spring, useVideoConfig} from 'remotion';
import {C, FONT} from './theme';

const N = 14;
const DROPPED = [4, 11];          // the two the model called "not a goal"

/** Flexbox does the layout, so chips and labels physically cannot collide --
 *  the bug that had to be fixed twice in the matplotlib version. */
const Row: React.FC<{
  chips: number;
  mode: 'filter' | 'label';
  progress: number;
}> = ({chips, mode, progress}) => (
  <div style={{display: 'flex', gap: 14, justifyContent: 'center'}}>
    {Array.from({length: chips}).map((_, k) => {
      const gone = mode === 'filter' && DROPPED.includes(k);
      const marked = mode === 'label' && !DROPPED.includes(k) && progress > 0.02;
      return (
        <div
          key={k}
          style={{
            width: 74,
            height: 96,
            borderRadius: 10,
            background: C.panel,
            border: `2px solid ${gone ? C.red : marked ? C.green : '#cdd5df'}`,
            opacity: gone ? 1 - progress : 1,
            transform: `scale(${gone ? 1 - progress * 0.25 : 1})`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {marked && (
            <div
              style={{
                width: 22,
                height: 22,
                borderRadius: '50%',
                background: C.green,
                opacity: Math.min(1, progress * 1.4),
              }}
            />
          )}
        </div>
      );
    })}
  </div>
);

export const LabelNotFilter: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const at = (s: number) => spring({frame: frame - s * fps, fps, durationInFrames: 20});
  const filterP = at(1.6);
  const markP = at(4.0);

  return (
    <AbsoluteFill
      style={{
        fontFamily: FONT,
        color: C.fg,
        padding: '56px 80px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        gap: 54,
      }}
    >
      <h1 style={{fontSize: 62, fontWeight: 700, textAlign: 'center', margin: 0,
                  letterSpacing: '-0.02em', opacity: at(0.1)}}>
        거른다 vs 표시한다
      </h1>

      <section style={{opacity: at(0.5), display: 'flex', flexDirection: 'column', gap: 14}}>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'baseline'}}>
          <span style={{color: C.red, fontSize: 32, fontWeight: 700}}>거르면</span>
          <span style={{color: C.mut, fontSize: 28, opacity: filterP}}>
            {N - DROPPED.length}개 남음
          </span>
        </div>
        <Row chips={N} mode="filter" progress={filterP} />
        <p style={{color: C.mut, fontSize: 26, textAlign: 'center', margin: 0, opacity: filterP}}>
          선방·아쉬운 슛까지 같이 버려진다
        </p>
      </section>

      <section style={{opacity: at(3.0), display: 'flex', flexDirection: 'column', gap: 14}}>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'baseline'}}>
          <span style={{color: C.green, fontSize: 32, fontWeight: 700}}>표시하면</span>
          <span style={{color: C.mut, fontSize: 28, opacity: markP}}>
            {N}개 유지 · {N - DROPPED.length}개 표시
          </span>
        </div>
        <Row chips={N} mode="label" progress={markP} />
      </section>

      <p style={{color: C.mut, fontSize: 26, textAlign: 'center', margin: 0,
                 opacity: interpolate(frame, [5 * 30, 5.5 * 30], [0, 1], {extrapolateRight: 'clamp'})}}>
        표시는 데이터에만 · 영상엔 그리지 않음
      </p>
    </AbsoluteFill>
  );
};

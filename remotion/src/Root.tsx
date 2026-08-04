import React from 'react';
import {Composition} from 'remotion';
import {LabelNotFilter} from './LabelNotFilter';
import {FPS} from './theme';

export const RemotionRoot: React.FC = () => (
  <Composition
    id="LabelNotFilter"
    component={LabelNotFilter}
    durationInFrames={7 * FPS}
    fps={FPS}
    width={1920}
    height={1080}
  />
);

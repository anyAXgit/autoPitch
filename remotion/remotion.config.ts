import {Config} from '@remotion/cli/config';

// Match the Python graphics: ProRes 4444 with a real alpha channel so the
// output drops straight onto a Final Cut timeline over the footage.
Config.setVideoImageFormat('png');
Config.setPixelFormat('yuva444p10le');
Config.setCodec('prores');
Config.setProResProfile('4444');

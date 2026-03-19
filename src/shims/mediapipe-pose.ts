// The package only ships a UMD bundle, so we load it for side effects and then
// re-export the global constructor that the bundle attaches in the browser.
import '../../node_modules/@mediapipe/pose/pose.js';

type GlobalPoseConstructor = {
  Pose?: unknown;
};

export const Pose = (globalThis as typeof globalThis & GlobalPoseConstructor).Pose;

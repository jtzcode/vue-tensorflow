import { Keypoint, Pose, PoseNet } from '@tensorflow-models/posenet';
import * as posenet from '@tensorflow-models/posenet';
import { DrawPose } from './DrawPose';
import * as tf from '@tensorflow/tfjs';

export class PoseClassifier {
    private model: PoseNet | null = null;
    private drawPose: DrawPose | null = null;

    constructor() {
        tf.ENV.set('WEBGL_PACK', false);
    }

    public async Pose(image: HTMLImageElement, canvas: HTMLCanvasElement): Promise<Keypoint[] | null> {
        if (!this.model) {
            this.model = await posenet.load();
            this.drawPose = new DrawPose(canvas);
        }

        if (this.model) {
            const result: Pose = await this.model.estimateSinglePose(image);
            if (result) {
                this.drawPose!.Draw(result.keypoints);
                return result.keypoints;
            }
        }
        return null;
    }

}
import { MobileNet } from '@tensorflow-models/mobilenet';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';


export interface TensorInformation {
    className: string;
    probability: number;
}

export class ImageClassifier {
    private model: MobileNet | null = null;
    constructor() {
        tf.ENV.set('WEBGL_PACK', false);
    }

    public async Classify(image: tf.Tensor3D | ImageData | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement): Promise<TensorInformation[] | null> {
        if (!this.model) {
            this.model = await mobilenet.load(); 
        }

        if (this.model) {
            const result = await this.model.classify(image);
            return {
                ... result,
            };
        }
        return null;
    }
}
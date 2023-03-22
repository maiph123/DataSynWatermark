import cv2
import argparse
from imwatermark import WatermarkEncoder, WatermarkDecoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm', type=str, default='watermark', help="watermark")
    parser.add_argument('--image', type=str, default='./Cloud.jpeg', help="path of image to be watermark")
    args = parser.parse_args()
    wm = args.wm
    img_path = args.image
    bgr = cv2.imread(img_path)
    print('Input watermark:', wm)
    # encode watermark
    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', wm.encode('utf-8'))
    bgr_encoded = encoder.encode(bgr, 'dwtDctSvd')
    cv2.imwrite('./Cloud_wm.jpeg', bgr_encoded)
    print("Watermarked image saved to: Cloud_wm.jpeg")
    # decode watermark
    bgr_encoded = cv2.imread('./Cloud_wm.jpeg')
    decoder = WatermarkDecoder('bytes', len(wm)*8)
    watermark = decoder.decode(bgr_encoded, 'dwtDctSvd')
    print("Detect watermark:", watermark.decode('utf-8'))
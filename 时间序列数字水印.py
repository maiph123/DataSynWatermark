import hashlib
import pandas as pd
import numpy as np
import argparse
import struct

def stringToBinary(string):
    return ''.join(format(ord(x), 'b') for x in string)


def decimalToBinary(n):
    return bin(n).replace("0b", "")


def binaryToDecimal(n):
    return int(n, base=2)


def changeLSB(n, lsb, wmb):
    # input binary string
    # change the bth least significant bit to wmb
    n = list(n)
    n[-(lsb + 1)] = str(wmb)
    n = ''.join(n)
    return n


def _hash(idx, key):
    # return the hash of value n using private key
    # obtain most significant bit
    out = hashlib.md5((idx + key).encode()).hexdigest()
    out = int(out, 16)
    return out


def detect(idx, n, key, wm):
    # add the detected bit to the corresponding place in wm
    wm_length = len(wm[0])
    binary = decimalToBinary(n)
    label_hash = hashlib.md5((idx + key).encode()).hexdigest()
    label_hash = int(label_hash, 16)
    lsb_idx = label_hash % max_lsb
    this_bit = binary[-(lsb_idx + 1)]
    wm_idx = label_hash % wm_length
    wm[int(this_bit), wm_idx] += 1
    return wm


def construct(wm):
    wm_length = len(wm[0])
    rec_wm = []
    for i in range(wm_length):
        if wm[0, i] > wm[1, i]:
            rec_wm.append(0)
        elif wm[0, i] < wm[1, i]:
            rec_wm.append(1)
        else:
            rec_wm.append(np.random.randint(2))
    return np.array(rec_wm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm', type=str, default='watermark', help="watermark")
    parser.add_argument('--pk', type=str, default='key', help="private key")
    parser.add_argument('--max_lsb', type=int, default=3, help="maximum number of least significant bits")

    args = parser.parse_args()
    # encode water to binary array
    print('Input watermark:', args.wm)
    watermark = str(args.wm).encode('utf-8')
    wmbits = np.array([n for n in watermark], dtype=np.uint8)
    wmbits = np.unpackbits(wmbits)
    private_key = args.pk
    max_lsb = args.max_lsb
    # 读入数据
    df = pd.read_csv('./small_drought_data.csv')
    ts_idx = (df.date.astype(str) + df.fips.astype(str)).values
    # 选取'PRECTOT'列进行水印嵌入
    ts_data = df['PRECTOT'].values
    ts_data = (ts_data * 100).astype(int)
    # 嵌入水印
    wm_length = len(wmbits)
    for i in range(len(ts_data)):
        this_v = ts_data[i]
        if abs(this_v) > 2 ** (max_lsb + 2):
            this_idx = ts_idx[i]
            this_vb = decimalToBinary(this_v)
            label_hash = _hash(this_idx, private_key)
            lsb_idx = label_hash % max_lsb
            wm_idx = label_hash % wm_length
            this_vb = changeLSB(this_vb, lsb_idx, wmbits[wm_idx])
            ts_data[i] = int(this_vb, base=2)
    # 输出加水印的数据
    df['PRECTOT'] = ts_data / 100
    df.to_csv('./small_drought_data_wc.csv', index=False)
    print('File saved to: small_drought_data_wc.csv')
    # 检测水印
    wm_vect = np.zeros((2, wm_length))
    for i in range(len(ts_data)):
        this_v = ts_data[i]
        if abs(this_v) > 2 ** (max_lsb + 1):
            this_idx = ts_idx[i]
            wm_vect = detect(this_idx, this_v, private_key, wm_vect)
    rec_wm = construct(wm_vect)
    rec_wm = np.packbits(rec_wm)
    bstr = b''
    for i in range(wm_length // 8):
        bstr += struct.pack('>B', rec_wm[i])
    rec_wm = bstr.decode('utf-8')
    print('Dectected watermark:', rec_wm)

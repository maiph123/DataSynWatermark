import numpy as np
import pandas as pd
import argparse
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfips', type=int, default=2, help="number of fips")
    parser.add_argument('--epochs', type=int, default=1000, help="number of epochs")

    args = parser.parse_args()
    nfips = args.nfips
    epochs = args.epochs
    # 数据读入
    drought_df = pd.read_csv('./drought_data.csv')
    drought_df['date'] = pd.to_datetime(drought_df['date'])
    selected_fips = drought_df.fips.unique()[:nfips]
    drought_df = drought_df[drought_df['fips'].isin(selected_fips)]
    drought_df = drought_df.sort_values(['fips', 'date'])
    drought_df = drought_df[drought_df['date'] != '2020/12/31']

    # 生成合成数据
    out_df = pd.DataFrame(columns=list(drought_df.drop(columns=['score', 'fips', 'date']).columns))
    out = np.array([])
    for fip in selected_fips:
        n = 2
        features = drought_df[drought_df['fips'] == fip]
        features = drought_df.drop(columns=['score', 'fips', 'date']).to_numpy()
        features = features.reshape(n, -1, features.shape[1])
        # Shape is now (# examples, # time points, # features)
        print('Train for fip:', fip)
        # Train DGAN model for first fip
        model = DGAN(DGANConfig(
            max_sequence_len=features.shape[1],
            sample_len=5,
            batch_size=min(1000, features.shape[0]),
            apply_feature_scaling=True,
            apply_example_scaling=False,
            use_attribute_discriminator=False,
            generator_learning_rate=1e-4,
            discriminator_learning_rate=1e-4,
            epochs=epochs,
            cuda=True
        ))

        model.train_numpy(features)

        # Generate synthetic data
        _, synthetic_features = model.generate_numpy(2)
        this_out = synthetic_features.reshape(-1, features.shape[-1])
        if len(out) == 0:
            out = this_out
        else:
            out = np.concatenate([out, this_out])

    # 输出合成数据csv
    out_df = pd.DataFrame(data=out,
                          columns=list(drought_df.drop(columns=['score', 'fips', 'date']).columns))
    out = synthetic_features.reshape(-1, features.shape[-1])
    out_df = pd.DataFrame(data=out,
                          columns=list(drought_df.drop(columns=['score', 'fips', 'date']).columns))
    out_df['score'] = drought_df['score']
    out_df['fips'] = drought_df['fips']
    out_df['date'] = drought_df['date']
    out_df.to_csv('./sim_drought_data.csv', index=False)
    print('File saved to: sim_drought_data.csv')
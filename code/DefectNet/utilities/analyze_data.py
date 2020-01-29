import seaborn as sns
import matplotlib.pyplot as plt
import json
from pandas.io.json import json_normalize
import os
import pandas as pd


def read_json(json_path):
    results = []
    np_names = [
        'AP', 'AP:0.50', 'AP:0.75', 'AP:S', 'AP:M', 'AP:L',
        'AR', 'AR:0.50', 'AR:0.75', 'AR:S', 'AR:M', 'AR:L',
    ]
    with open(json_path) as fp:
        lines = fp.readlines()
        for line in lines:
            r = json.loads(line)
            d = r['data']['bbox']['data']
            result = dict(cfg=r['cfg'], mode=r['mode'])
            for k, v in zip(np_names, d['coco_eval']):
                result[k] = v
            for k, v in d['classwise'].items():
                result[k] = v
            results.append(result)
    return results


def phrase_json(json_path):
    save_path = json_path[:-5] + '.csv'
    if os.path.exists(save_path):
        return pd.read_csv(save_path)
    results = read_json(json_path)
    df = json_normalize(results)
    df.to_csv(save_path, index=False)
    return df


def main():
    data = phrase_json('../config_alcohol/cascade_rcnn_r50_fpn_1x/eval_alcohol_dataset_report.json')

    sns.set(style="darkgrid")
    ids = []
    for i in range(data.shape[0]):
        r = data.iloc[i]
        arrs = r['cfg'].split('_')
        ids.append(float(arrs[1][:-1]))

    x_name, y_name = 'Times of learning rate', 'Average Precision'
    data[x_name] = ids
    data = data.rename(columns={'AP': y_name})
    ax = sns.lineplot(
        x=x_name, y=y_name,
        hue="mode",
        style="mode",
        markers=True,
        dashes=False,
        data=data,
        ci=None
    )
    plt.savefig('../results/imgs/AP-times_of_lr.jpg')
    plt.show()


if __name__ == '__main__':
    main()

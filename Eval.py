# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import sys
import cv2
import json
import argparse

sys.path.append("..")
import py_sod_metrics

def Eval_COD_SOD_Num1(mask_root, pred_root):
    # 核心指标计算部分没有任何修改
    FM = py_sod_metrics.Fmeasure()
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()

    sample_gray = dict(with_adaptive=True, with_dynamic=True)
    sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
    overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)
    FMv2 = py_sod_metrics.FmeasureV2(
        metric_handlers={
            # 灰度数据指标
            "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
            "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.1),
            "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
            "rec": py_sod_metrics.RecallHandler(**sample_gray),
            "iou": py_sod_metrics.IOUHandler(**sample_gray),
            "dice": py_sod_metrics.DICEHandler(**sample_gray),
            "spec": py_sod_metrics.SpecificityHandler(**sample_gray),
            "ber": py_sod_metrics.BERHandler(**sample_gray),
            "oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
            "kappa": py_sod_metrics.KappaHandler(**sample_gray),
            # 二值化数据指标的特殊情况一：各个样本独立计算指标后取平均
            "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
            "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_bin, beta=1),
            "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_bin),
            "sample_birec": py_sod_metrics.RecallHandler(**sample_bin),
            "sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
            "sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),
            "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_bin),
            "sample_biber": py_sod_metrics.BERHandler(**sample_bin),
            "sample_bioa": py_sod_metrics.OverallAccuracyHandler(**sample_bin),
            "sample_bikappa": py_sod_metrics.KappaHandler(**sample_bin),
            # 二值化数据指标的特殊情况二：汇总所有样本的tp、fp、tn、fn后整体计算指标
            "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
            "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_bin, beta=1),
            "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_bin),
            "overall_birec": py_sod_metrics.RecallHandler(**overall_bin),
            "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
            "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
            "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
            "overall_biber": py_sod_metrics.BERHandler(**overall_bin),
            "overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
            "overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
        }
    )

    mask_name_list = sorted(os.listdir(mask_root))
    for i, mask_name in enumerate(mask_name_list):
        # 优化了一下打印，减少刷屏，可以自己改回原来的
        if i % 50 == 0 or i == len(mask_name_list) - 1:
            print(f"[{i}/{len(mask_name_list)}] Processing {mask_name}...")
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)
        FMv2.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]
    fmv2 = FMv2.get_results()

    curr_results = {
        "MAE": mae,
        "Smeasure": sm,
        "wFmeasure": wfm,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),
        "adpfm": fmv2["fm"]["adaptive"],
        "meanfm": fmv2["fm"]["dynamic"].mean(),
        "maxfm": fmv2["fm"]["dynamic"].max(),
        "sample_bifm": fmv2["sample_bifm"]["binary"],
        "overall_bifm": fmv2["overall_bifm"]["binary"],
        "adppre": fmv2["pre"]["adaptive"],
        "meanpre": fmv2["pre"]["dynamic"].mean(),
        "maxpre": fmv2["pre"]["dynamic"].max(),
        "sample_bipre": fmv2["sample_bipre"]["binary"],
        "overall_bipre": fmv2["overall_bipre"]["binary"],
        "adprec": fmv2["rec"]["adaptive"],
        "meanrec": fmv2["rec"]["dynamic"].mean(),
        "maxrec": fmv2["rec"]["dynamic"].max(),
        "sample_birec": fmv2["sample_birec"]["binary"],
        "overall_birec": fmv2["overall_birec"]["binary"],
        "adpdice": fmv2["dice"]["adaptive"],
        "meandice": fmv2["dice"]["dynamic"].mean(),
        "maxdice": fmv2["dice"]["dynamic"].max(),
        "sample_bidice": fmv2["sample_bidice"]["binary"],
        "overall_bidice": fmv2["overall_bidice"]["binary"],
        "adpiou": fmv2["iou"]["adaptive"],
        "meaniou": fmv2["iou"]["dynamic"].mean(),
        "maxiou": fmv2["iou"]["dynamic"].max(),
        "sample_biiou": fmv2["sample_biiou"]["binary"],
        "overall_biiou": fmv2["overall_biiou"]["binary"],
        "adpf1": fmv2["f1"]["adaptive"],
        "meanf1": fmv2["f1"]["dynamic"].mean(),
        "maxf1": fmv2["f1"]["dynamic"].max(),
        "sample_bif1": fmv2["sample_bif1"]["binary"],
        "overall_bif1": fmv2["overall_bif1"]["binary"],
        "adpspec": fmv2["spec"]["adaptive"],
        "meanspec": fmv2["spec"]["dynamic"].mean(),
        "maxspec": fmv2["spec"]["dynamic"].max(),
        "sample_bispec": fmv2["sample_bispec"]["binary"],
        "overall_bispec": fmv2["overall_bispec"]["binary"],
        "adpber": fmv2["ber"]["adaptive"],
        "meanber": fmv2["ber"]["dynamic"].mean(),
        "maxber": fmv2["ber"]["dynamic"].max(),
        "sample_biber": fmv2["sample_biber"]["binary"],
        "overall_biber": fmv2["overall_biber"]["binary"],
        "adpoa": fmv2["oa"]["adaptive"],
        "meanoa": fmv2["oa"]["dynamic"].mean(),
        "maxoa": fmv2["oa"]["dynamic"].max(),
        "sample_bioa": fmv2["sample_bioa"]["binary"],
        "overall_bioa": fmv2["overall_bioa"]["binary"],
        "adpkappa": fmv2["kappa"]["adaptive"],
        "meankappa": fmv2["kappa"]["dynamic"].mean(),
        "maxkappa": fmv2["kappa"]["dynamic"].max(),
        "sample_bikappa": fmv2["sample_bikappa"]["binary"],
        "overall_bikappa": fmv2["overall_bikappa"]["binary"],
    }
    return curr_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SOD/COD Predictions")
    parser.add_argument("--mask_root", type=str, required=True, help="GT")
    parser.add_argument("--pred_root", type=str, required=True, help="Pre")
    parser.add_argument("--save_path", type=str, required=True, help="Save JSON")
    
    args = parser.parse_args()

    # 计算指标
    curr_results = Eval_COD_SOD_Num1(mask_root=args.mask_root, pred_root=args.pred_root)
    
    # 确保保存目录存在
    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    # 写入 JSON
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(curr_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n[Success] 评估结果已保存至: {args.save_path}\n")
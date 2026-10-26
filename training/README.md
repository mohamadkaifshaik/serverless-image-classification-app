# training

`train.py` does transfer learning with MobileNetV2 on the `tf_flowers` dataset
(daisy, dandelion, roses, sunflowers, tulips).

```
python training/train.py                    # full run
python training/train.py --quick            # smoke test (~1 min)
python training/train.py --finetune-layers 30 --finetune-epochs 12
```

The dataset is downloaded to `~/.keras/datasets` on first run.

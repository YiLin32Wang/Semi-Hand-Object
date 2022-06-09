# Bug mark note
1. dataset报错
    ```Shell
        Traceback (most recent call last):
    File "traineval.py", line 104, in <module>
        main(args)
    File "traineval.py", line 35, in main
        train_dat = get_dataset(args, mode="train")
    File "/home/yilin/Hand_est/Semi-Hand-Object/utils/utils.py", line 133, in get_dataset
        train_label_root="ho3d-process", mode=mode, inp_res=args.inp_res)
    File "/home/yilin/Hand_est/Semi-Hand-Object/dataset/ho3d.py", line 55, in __init__
        self.set_list = ho3d_util.load_names(os.path.join(train_label_root, "train.txt"))
    File "/home/yilin/Hand_est/Semi-Hand-Object/dataset/ho3d_util.py", line 14, in load_names
        with open(image_path) as f:
    FileNotFoundError: [Errno 2] No such file or directory: 'ho3d-process/train.txt'
    ```
1. 使用了3 gpu，然而当前资源条件只能使用2 gpu

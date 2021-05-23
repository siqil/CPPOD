for dataset in pois gam
do
    python train_cppod.py --dataset $dataset --label-size 3
    python test_cppod.py --dataset $dataset --label-size 3
    python train_cppod.py --dataset $dataset --label-size 3 --noncontext
    python test_cppod.py --dataset $dataset --label-size 3 --noncontext
done

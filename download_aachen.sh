

# get inputs from the user
while getopts "d:" OPTION; do
    case $OPTION in
        d) DATA_ROOT=$OPTARG;;
        *) exit 1 ;;
    esac
done

# set data root default
if [ "$DATA_ROOT" ==  "" ];then
        DATA_ROOT=$HOME/datasets/
fi

mkdir -p $DATA_ROOT/
ln -fs $DATA_ROOT data
mkdir -p $DATA_ROOT/aachen/

DRIVE_ID=18vz-nCBFhyxiX7s-zynGczvtPeilLBRB
URL=https://drive.google.com/uc?id=$DRIVE_ID
gdown $URL -O $DATA_ROOT/aachen/database_and_query_images.zip
bash download_r2d2_training_data.sh 
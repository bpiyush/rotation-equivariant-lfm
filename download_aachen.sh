

# get inputs from the user
while getopts "d:" OPTION; do
    case $OPTION in
        d) data_root=$OPTARG;;
        *) exit 1 ;;
    esac
done

# set data root default
if [ "$data_root" ==  "" ];then
       data_root=$HOME/datasets/
fi

mkdir -p $data_root/

pre=${1}
DIR="../work_dirs/breast/cascade_rcnn_x101_64x4d_fpn_1x_${pre}+multiscale+softnms"
TEST_DIR="$HOME/undone-work/data/detection/breast/annotations/"
python do_submit.py ${TEST_DIR}/test_data_A.json "${DIR}/data_mode=test+.bbox.json" ${DIR}/submit_nobg_${pre} --convert filter_id=0 cvt_img_id=json cvt_box=xywh2xyxy cvt_score=append cvt_cat_id=True
cd $DIR
zip -q -r submit_nobg_${pre}.zip submit_nobg_${pre}
echo "OK!"

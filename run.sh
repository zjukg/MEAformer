# # w/ surface
# DBP15K
bash run_meaformer.sh 0 DBP15K zh_en 0.3 1 
bash run_meaformer.sh 0 DBP15K ja_en 0.3 1 
bash run_meaformer.sh 0 DBP15K fr_en 0.3 1 

# # w/o surface
# DBP15K
bash run_meaformer.sh 0 DBP15K zh_en 0.3 0 
bash run_meaformer.sh 0 DBP15K ja_en 0.3 0 
bash run_meaformer.sh 0 DBP15K fr_en 0.3 0
# FBYG15K
bash run_meaformer.sh 0 FBYG15K norm 0.8 0 
bash run_meaformer.sh 0 FBYG15K norm 0.5 0 
bash run_meaformer.sh 0 FBYG15K norm 0.2 0 
# FBDB15K
bash run_meaformer.sh 0 FBDB15K norm 0.8 0 
bash run_meaformer.sh 0 FBDB15K norm 0.5 0 
bash run_meaformer.sh 0 FBDB15K norm 0.2 0 












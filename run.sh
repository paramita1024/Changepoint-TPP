for i in 1 #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 
do
      for partition in  1000 # 1500 2500 750 1200 2000 
      do
            python3 -m change_point_detection.exec_change_point_detection  --method Score_statistics --gamma 1000 --window_length 20 --seed $i --num_changepoints 1 --dim_inner_of_THP 16 
            python3 -m change_point_detection.exec_change_point_detection  --method GLR_Hawkes --gamma 1000 --window_length 20 --seed $i --num_changepoints 1 --dim_inner_of_THP 16 
            python3 -m change_point_detection.exec_change_point_detection  --method Greedy_selection  --gamma 1000 --window_length 20 --seed $i --num_changepoints 1 --dim_inner_of_THP 16 
            python3 -m change_point_detection.exec_change_point_detection --method differentiable_change_point_detector --seed $i --num_changepoints 1  --save_interval 1000 --dim_inner_of_THP 16  --pre_train_CPD_model 
            python3 -m change_point_detection.exec_change_point_detection --method differentiable_change_point_detector --seed $i --num_changepoints 1  --save_interval 1000  --dim_inner_of_THP 16 --partition_method cvxpy --perturb --load_pre_trained_CPD_model 
            python3 -m change_point_detection.exec_change_point_detection --method differentiable_change_point_detector --seed $i --window_length 30 --gamma 1000 --num_changepoints 1 --whether_global --save_interval 1000  --dim_inner_of_THP 16 --partition_method cvxpy --perturb  
      done
done


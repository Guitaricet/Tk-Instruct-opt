 Corruptions
 empty-baseline1 -  x . 
                   # set --add_task_definition False \
                   #     --num_pos_examples 0 \
                   #     --num_neg_examples 0 \
                   #     --add_explanation False \
                   #     --corruption baseline1
 examples-baseline2 -  xy xy xy xy | x
                   # set --add_task_definition False \
                   #     --num_pos_examples 4 \
                   #     --num_neg_examples 0 \
                   #     --add_explanation False \
                   #     --corruption baseline2
                   
 1) instr-placement-before-ex -   I | xy xy xy xy | x
 2) instr-placement-after-ex  -       xy xy xy xy | I | x
 3) instr-randomwords        -   I(random) | xy xy xy xy | x 
 4) instr-frequentwords      -   I(freq) | xy xy xy xy | x 
 Rethinking corruptions 
 5) label-random-labelspace        -   I | xy(ran) xy(ran) xy(ran) xy(ran) | x  # works only for datasets where there is label options and all are incorect labels
 6) label-random-labelspace-half   -   I | xy(ran) xy(ran) xy xy | x  # works only for datasets where there is label options, every other label is incorrect
 7) label-randomwords              -   I | xy(ran) xy(ran) xy(ran) xy(ran) | x # label space is created with random words and y i s randomly chosen from this label space
 8) label-empty                    -   I | x x x x | x   # Input:x
 9) input-empty                   -   I | y y y y (number of classes) | x  # Output:y
 10) input-oodrandom                -   I | x(oodran)y x(oodran)y x(oodran)y x(oodran)y | x \TODO get the common crawl corpus
 for all these no explanation and negative examples are added

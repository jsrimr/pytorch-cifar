if cur_block != None:
            # 처음 : proj c_out + next_exp_c_in
            # 중간 : exp_cout ~ proj c_in
            # 마지막 : next_exp_c_out ~ proj c_out
            stage_block_weights = []
            for block in range(last_block + 1):
                flag = False
                new_block_weights = []
                if block != 0:
                    # exp c_out ~ proj c_in, bn3
                    parent_block_weights = get_collection(parent_cc._name2val, set(parent_cc._name2val.keys()), stage,
                                                          block)
                    parent_block_weights[0] = new_weights[-1]  # get wider c_in exp_layer

                    child_block_weights = get_collection(child_cc._name2val, set(child_cc._name2val.keys()), stage,
                                                         block)
                    next_block_weight = get_next_expansion_weight(parent_cc._name2val, stage, block,
                                                                  is_last_block=(block == last_block))
                    child_next_block_weight = get_next_expansion_weight(child_cc._name2val, stage, block,
                                                                        is_last_block=(block == last_block))
                    new_block_weights = wider_MBconv_block(parent_block_weights, child_block_weights,
                                                           next_block_weight,
                                                           child_next_block_weight)
                    flag = True  #

                # NOTE : 이러면 block0 의 경우는 expansion 이랑 depthwise 부분의 width 가 안늘어날거 같은데? => 그게 맞음. exp 의 width 는 그냥 *6 으로 결정될 뿐이고, depth 도 그걸 따라가기 때문
                # proj c_out, bn3 ~ next exp c_in늘림
                child_proj_and_bn3_weights = get_collection(child_cc._name2val, set(child_cc._name2val.keys()),
                                                            stage,
                                                            block)[-2:]
                parent_proj_and_bn3_weights = get_collection(parent_cc._name2val, set(parent_cc._name2val.keys()),
                                                             stage,
                                                             block)[-2:]
                if flag:  # 위의 로직이 실행되어 proj c_in 이 늘어난 경우를 처리
                    parent_proj_and_bn3_weights[0] = new_block_weights[-2]
                child_next_expansion_weight = get_next_expansion_weight(child_cc._name2val, stage, block,
                                                                        is_last_block=(block == last_block))
                parent_next_expansion_weight = get_next_expansion_weight(parent_cc._name2val, stage, block,
                                                                         is_last_block=(block == last_block))
                new_weights = proj_wider_cout_expansion_wider_cin(child_proj_and_bn3_weights,
                                                                  parent_proj_and_bn3_weights,
                                                                  child_next_expansion_weight,
                                                                  parent_next_expansion_weight)
                if new_block_weights:
                    new_block_weights[-2:] = new_weights[:2]
                else:
                    new_block_weights = new_weights[:2]

                stage_block_weights.extend(new_block_weights)

            # 마지막 : next_exp_c_out ~ proj c_out => proj_cout 은 왜 신경쓰는거지??
            parent_block_weights = get_collection(parent_cc._name2val, set(parent_cc._name2val.keys()),
                                                  stage + 1,
                                                  block=0)
            parent_block_weights[0] = new_weights[-1]  # get wider c_in exp_layer
            child_block_weights = get_collection(child_cc._name2val, set(child_cc._name2val.keys()),
                                                 stage + 1,
                                                 block=0)
            new_block_weights = wider_MBconv_block(parent_block_weights, child_block_weights,
                                                   next_stage_first_block=True)
            stage_block_weights.extend(new_block_weights)

            head_keys = [k for k in filtered_keys if
                         ('model/head' in k) or ('model/mb_v3compat_json_net/head/' in k) or (
                                 'model/mb_v3compat_json_net/dense/' in k)]
            if head_keys:  # width_stage5 인 경우에는 head 도 바뀜
                new_head_weights = wider_head_weights(parent_cc, child_cc, head_keys)
                stage_block_weights.extend(new_head_weights)
# PhyEditing 收尾验证（2026-10-08）

已知参考初速度的调优筛选集达到目标；未知初速度及低帧率尚未全面达到最大误差要求。支持优于已测基线，不能证明全局最优。

|反演口径|成功拟合|平均误差|最大误差|
|---|---:|---:|---:|
|已知初速度|1038 / 1038|1.11%|6.2%|
|联合估计初速度，30 fps|1038 / 1038|2.88%|19.8%|
|联合估计初速度，15 fps|845 / 1038|3.17%|34.2%|

门槛在本批数据调优，尚无冻结后的独立测试。仍使用参考窗口、位置和姿态。背景重复必须按物理运动组划分以避免泄漏。

## 六模型生成测试

每模型40条。固定窗口，联合估计初速度。误差为 abs(g估计/g目标-1)；无法跟踪计入总分母；触及边界的拟合仅作诊断。

|模型|记录|可拟合含边界|边界|无法跟踪|误差≤20% / 全40条|拟合误差中位数|
|---|---:|---:|---:|---:|---:|---:|
|LTX-Video-2B|40|37|23|3|0 / 40|93.8%|
|Wan2.2-TI2V-5B|40|31|23|9|0 / 40|98.2%|
|CogVideoX1.5-5B|40|30|15|10|0 / 40|95.6%|
|Cosmos3-Nano|40|35|25|5|0 / 40|97.3%|
|Wan2.2-I2V-A14B|40|39|23|1|2 / 40|95.0%|
|PhysAlign（Wan2.2-A14B）|40|38|16|2|1 / 40|93.8%|

整段滑窗任意命中是多次尝试的宽松诊断，不能作主成功率。换目标命中不是统计显著性。地球参照只匹配部分条目。

已知初速度口径：信息对齐2D平均3.69%；真值深度抬升1.91%；六深度模型约20–30%。成功条目均值须和失败数一起报告，详见 report.html 及 runs/final_report.json。各方法可见帧使用存在差异，须补充完全相同观测子集的对比。

优先完成独立运动组测试、未知初速度调优及低帧率尾部误差修复，再决定训练。六模型生成—跟踪—评分链路已输出结果，生成质量未达到目标；尚未训练。

来源：runs/benchmark_gravity_v1/items.jsonl、runs/final_report.json、runs/genfloor_pred_*_agnostic.jsonl、各生成模型 rows/pred/scan 文件。复算工具 scripts/phyediting_gen_summary.py；用 RUN_DIR:MODEL 参数指定模型。

## 新增：共同观测复算

新增：完全相同观测帧的对比
1,036 / 1,038 条保留至少四个完整可见、未裁切帧；关闭我们独有的接触截断，确保实际使用的帧一致。均值仅计算成功条目，失败和缺失另列。仍是调优集。

完整统计见 common_observations.json。我们均值1.19%、最大15.62%；2D均值3.10%；预测深度20.44–35.09%；真值深度0.87%。旧表不再用于证明所有方法中最优。

## 本轮调优全量结果

共同观测、已知初速度：centred 平均0.89%、最大8.99%，1036条全部成功。未知初速度：30 fps 平均2.64%、最大21.06%；15 fps 845/1038成功、平均2.93%、最大28.34%。虽然均值改善，仍未全面达到最大误差要求，因此默认保持anchor。没有按结果删除难例。详情见 offset_comparison.json，包含事件、重力和物理组等权统计。生成评分默认改为agnostic以避免参考初速度泄漏。

## 尾部误差根因诊断

30/15fps各选10个已有最差案例，只作根因诊断。换用compact仿真真实位置和姿态投影出的轮廓框、相同时刻、未知初速度，centred误差全部小于0.2%；SAM2观测中部分超过20%。支持优先修观测与跟踪，不代表新的总体性能或独立测试。原始诊断见tail_diagnosis.json；脚本phyediting_tail_diagnosis.py。


## 亚像素观测控制（开发小试验）

10条视频中5条来自已有误差尾部，5条由固定哈希选择。对同一次SAM2 logits分别提取整数框与亚像素框，保持帧索引、未知初速度约束和centred损失相同。30fps成功10/10，最大误差20.42%→20.49%；15fps成功8/10，最大27.97%→21.93%。不能据此声称总体误差改善或达到最大10%目标；默认仍用整数框。逐队列和逐条见subpixel_pilot.json。


## 2026-10-08 逐边观测诊断

scripts/phyediting_edge_diagnosis.py对既有20条误差选中案例，在原始实际帧上比较SAM2框和仿真轮廓，去除常数与线性趋势，再记录曲率、残差RMS和时间相关性。真值只用于诊断，未提供给拟合器，也未据此删帧或改预测。完整结果edge_diagnosis.json。

T19三个记录的左右边曲率幅度约12–36px，底边仅0.17–0.40px；其余案例也存在亚像素到数像素曲率偏差。该量是去线性趋势后误差的二次投影幅度，不等于重力误差，不能证明所有尾部均为同一成因。尚未修改算法默认。没有新GPU任务，旧10视频任务已结束；本地只有报告HTTP服务。

## 固定边界消融（10视频开发试验）

原整数框、相同真实帧、未知初速度、centred偏移；所有组关闭接触截断，避免轴消融与截断混杂。全边/仅纵向/仅横向是统一规则，不按目标误差选边。30fps：全边均值7.805%、最大20.424%、10/10成功；仅纵向均值9.259%、最大22.875%、10/10成功；仅横向9/10成功、均值44.478%。哈希对照30fps均值由全边0.609%变为仅纵向2.643%。15fps：全边8/10成功、最大15.749%；仅纵向8/10成功、最大20.891%。单轴拟合损失信息，未改默认。详细edge_ablation.json；分难例/哈希对照与状态均保留。

关闭截断与原先开启截断的15fps全边小试验差异提示需要单独检验截断影响，不代表全量已改善。scripts/phyediting_offset_trial.py新增--no-contact-check；scripts/phyediting_contact_comparison.py拒绝部分数据，比较全部输入且统计实际拟合帧变化、事件/重力/相机/物理组的覆盖与失败。

## 2026-10-08 接触截断全量15fps控制完成

全部1038条开发样本，无误差删选，未知初速度centred拟合；输入观测与参考窗口相同，仅关闭接触逻辑。开启：845成功、193 no_track、均值2.928%、最大28.344%；关闭：845成功、193 no_track、均值2.917%、最大24.654%；覆盖81.41%，≤10%条数830→831，≤20%仍844/1038。实际拟合帧数改变50条。物理组等权成功均值2.914%→2.899%。完整各事件/重力/相机/物理组尝试数、失败、覆盖率见contact_comparison15.json。仍为已调优集，不是独立验证。结果未达尾部目标，不改默认。

本轮29测试通过、1跳过；本地试验已退出，无新GPU任务。下一步：先在低帧率尾部量化g与未知速度的雅可比耦合及同观测可辨识性，寻找可靠的观测增强/多物体共享约束。不要再扩展单轴弃边或亚像素重跟踪；保留193条帧不足在总分母。新方法应在固定开发试验验证后冻结，再补未知初速度公平基线与未调优物理设置测试。

## 2026-10-08 未知速度局部可辨识性诊断

scripts/phyediting_identifiability.py读取15fps已完成的contact-off centred拟合，在原观测帧上对log(g)和自由速度参数作有限差分；将重力雅可比列投影到速度参数列的正交补，避免固定速度的sensitivity夸大独立信息。无需重拟合，目标重力仅在诊断完成后用于误差分组，未改变任何预测、门槛或样本。

1038条全部保留：845成功、193 no_track。成功样本的独立重力信息比例中位数5.38%，条件雅可比范数中位数20.68px。误差>10%的14条比例中位数4.92%、范数14.97px；误差<3%的509条为5.44%、23.84px。比例组间接近，不能据此单阈值拒识或断言所有尾部均由不可辨识性造成。完整记录和事件/重力/相机/物理组统计见identifiability15.json。

这里只是拟合点附近、未作soft-L1曲率修正的局部线性诊断；unit_iid_px_logg_scale假设独立同方差1px噪声，不是经过校准的置信区间，不能保证最大误差。参数边界、非线性、分割相关误差尚未覆盖。没有新GPU任务和运行中的实验，默认算法未变。

下一步优先实现固定开发案例的局部重力profile：固定g、重新优化未知速度，检验尾部是否存在平坦或多个代价谷；结合观测残差确定是否需要多物体共享g约束。不得把真值速度作为先验，不根据目标误差逐条挑选谷。共享g需要明确额外物体信息并为基线匹配。帧不足193条仍留在覆盖分母；之后冻结协议并测未调优设置。

## 2026-10-08 重力profile控制完成

15fps contact-off centred中所有14条误差>10%开发案例，加5条固定SHA256对照；固定g，再对全部8个方向假设、各两组速度初值重新优化速度。网格为既有估计的0.5–1.8倍（含既有估计），未以真值为中心；目标g只在worker结束后追加用于诊断。共19条完整输出gravity_profile15.json，图gravity_profile15.svg。真值参与难例选择，因此不能作为独立性能或全局最优证明。

19条网格最优g均等于原估计，最大相对代价下降6.39e-12，拟合误差没有改变。在已检验网格与初值范围内，未发现求解器选错更低代价谷的证据；网格、方向假设、边界仍有限，不能断言全局不存在其他解。31测试通过、1跳过，未改默认算法；全部实验已结束，无新GPU任务。

下一步重点转回观测增强：先检查同一视频的额外物体跟踪与可用自由飞行窗口，设计共享g且各物体速度独立的配对试验；若没有同时可用观测，明确覆盖限制。参考速度和真实g不得进入共享拟合或窗口择优。额外物体/几何信息必须同步提供给同信息基线。固定开发规则后验证未调优物理设置；不扩大无收益的初值、单轴或亚像素试验。

## 2026-10-08 多物体共享g开发控制完成

已有tracks_all原始SAM2观测及all_oracle声明窗口审计：1038条中269条（25.92%）存在同时自由飞行、15fps至少4个可用真实观测帧的额外物体。按帧数与ID固定排序选择辅助窗口，不用真值g或目标误差择优；保留声明几何、姿态、可见边与参考飞行窗口。15fps contact-off误差>10%的14条均无此类辅助窗口。覆盖统计joint_coverage15.json包含事件/重力/相机/物理组；不能声称不存在其他尚未建索引的可用观测。

新增实验joint_fit：g共享、各对象速度独立，边界仍0–8m/s、全方向；centred损失、无接触截断。初值来自独立观测拟合与固定G_SEEDS，truth不参与拟合。每个对象方向假设固定为独立拟合胜出假设，联合阶段多初值优化，因此仍是实验求解器，不是全局最优保证。额外物体信息与旧单物体基线不匹配，不能直接宣称更优。

24固定SHA256开发pilot：单物体均值2.209%、最大6.721%；联合2.353%、8.210%，全部成功。269条完整辅助可用集：单物体均值2.567%、最大9.686%；联合2.426%、最大14.682%，全部成功。按固定规则“有辅助就联合、否则单物体”在1038条总体：845成功、193 no_track，均值2.917%→2.872%，最大仍24.654%，≤10%条数831→830，≤20%仍844。逐条joint_full15.json，分组joint_comparison15.json。没有按误差决定是否启用联合；默认仍不变。

33测试通过、1跳过。全部本地实验已结束，无新GPU任务。脚本后续可用--out区分pilot与全量文件，避免覆盖；24条原pilot已保留，269条另存joint_full15，运行期临时重用pilot路径已归档。复跑命令：phyediting_joint_coverage.py；phyediting_joint_trial.py --limit 0 --out joint_full15；phyediting_joint_report.py。

下一步：单物体尾部仍是优先项，先检查原视频轮廓与SAM2观测的局部运动/背景分离能否提供独立约束；非同时的辅助飞行窗口可另做覆盖审计，但不要继续盲目扩大同时时窗共享g。任何图像观测改进必须只用视频和声明条件，不能用真值轮廓修正；固定开发规则并控制同真实帧，在未调优物理设置上验证。低帧率193条帧不足仍计覆盖，未知速度公平2D及深度基线仍待完成。

## 2026-10-08 视频运动轮廓开发试验完成

19条既有profile诊断视频均可在本地读取。新增scripts/phyediting_motion_refine.py：仅用原15fps窗口内相同观测帧的图像中位背景，在SAM2框固定8px扩展ROI内取强度差>12的连通轮廓；固定面积/重叠门槛、不接受内部ROI裁切边界，未通过门槛回退原框。拟合未知速度、centred、无接触截断；不读真值轮廓、位置轨迹或目标g来改框。选择仍为14已有高误差+5固定哈希对照，不能作为总体成绩。

19条均成功拟合，原均值10.109%、最大24.654%；修正均值10.953%、最大53.013%。固定哈希5条原均值2.204%、最大4.641%；修正3.205%、7.999%。详细motion_refine15.json保留每帧修正框与回退原因。这种简单中位背景方法不可靠，不能替换默认。没有按真值误差回退或删帧，失败方案结果照实保留。

测试发现初版接受ROI截断连通分量，已按图像边界条件修复并重新完成19条试验；报告只包含修复后的结果。34测试通过、1跳过。实验已结束，无新GPU占用。

下一步优先补齐未知初速度下的同信息、同实际帧2D与六深度基线：窗口/相机/初始几何一致，不给参考速度，失败和物理组分组都保留。完成后再确定更有依据的观测改进，避免继续无收益的亚像素、单轴、同时辅助和简单背景阈值扫描。已有未调优物理设置应在固定协议后检验，不把已检查数据重新切分伪装未见测试。核心平均/尾部目标尚未全面达到。

## 2026-10-09 未知初速度共同拟合帧对比完成

复用既有1036条共同完整可见且未裁切观测，覆盖分母仍1038。新增phyediting_unknown_common.py移除v0与方向种子，速度0–8m/s、方向自由；centred、关闭接触截断。基线新增pixel2d_agnostic：方向已声明，图像速度作为自由线性项；深度抬升已有自由速度二次拟合。所有深度方法缺失任一拟合帧的有效深度即失败，不静默使用更少帧。没有重复运行GPU深度模型。

发现56条首个共同帧晚于声明初始帧。未知速度深度锚定因此用相同保留帧的深度二次拟合外推到声明时刻，再匹配声明表面深度，避免错误地把初始深度直接赋给晚帧。这是开发基线尺度估计，不是真值修正；旧已知速度结果不改。新增missing-depth与晚帧外推测试。

30fps开发对比：我们1036成功，均值2.909%、最大26.482%；2D1036成功，5.718%、41.949%；额外真值深度1036成功，2.260%、25.924%。六预测深度锚定：DepthPro741成功、54.604%；DA-V2 840、52.296%；ZoeDepth755、43.744%；MoGe-2 907、61.263%；UniDepthV2 815、59.523%；VGGT1006、43.010%。两条共同帧不足仍计入每方法总分母；完整失败、分组、raw尺度诊断及成功交集上的我们均值见unknown_common.json。不能把这些数字当作15fps尾部改善。

公平性仍有未完成项：2D/深度使用无范数上界的自由线性速度，我们使用0–8m/s先验；给出的声明相同，但表示/先验不是完全相同。我们显式模拟几何与自旋，抬升方法未利用同一前向模型。VGGT缓存推理可能使用共同拟合帧之外的帧，严格神经输入帧对齐尚须专门重跑；目前只是拟合帧相同。真值深度是额外oracle信息，均值优于我们，不得声称所有方法中全局最优。所有数据已检查，尚无独立验证。

实验已结束，无新GPU占用。下一步先消除上述先验/神经输入帧差异：实现与我们相同速度上界的2D和抬升控制，固定同观测协议再安排VGGT共同帧推理。继续保留失败分母，不按误差选尺度或速度范围。然后结合公平对比与独立新物理设置评估调优，不再重复无收益轮廓小试验。最大误差≤10%目标仍未实现。

## 2026-10-09 有界未知速度2D控制完成

新增bounded_motion控制：速度范数0–8m/s、仰角±89°、重力0.1–80，未知方向与速度；soft-L1残差，SLSQP多初值并明确记录不收敛/边界。函数不接收真值或参考速度。共同1036拟合帧保持不变，1038总分母中两条共同帧不足仍保留。完成结果平均5.863%、最大39.059%，1036成功；我们相同共同帧centred为2.909%、26.482%。事件/重力/相机/物理组尝试数与失败见bounded_2d.json。未改变我们的默认或宣称尾部已达标。

这是线性投影与中心抛物线模型的速度边界控制；它没有我们同样的旋转几何、阻力表示，有限初值不保证全局最优。SLSQP内部步骤出现边界裁切提示，返回解经过约束与收敛检查，未静默接受不合格解。新增合成未知速度恢复与速度范数约束测试通过。实验已结束，无新GPU占用。下一步为深度抬升加同样先验，并核实VGGT共同帧推理资源；不重复已有任务。

## 2026-10-09 六深度有界速度控制完成

复用相同1036条共同拟合帧，六模型共6216次，未重新运行GPU推理。深度完整性、初始深度外推尺度与上一轮一致；世界坐标抬升轨迹拟合速度范数0–8m/s、仰角±89°、重力0.1–80，函数不接受参考速度或真值g。所有模型仍以1038条为覆盖分母，缺帧、无共同帧、无有效尺度、边界与优化失败单列。

成功数/成功均值：DepthPro736/54.328%；DA-V2 830/51.781%；ZoeDepth750/43.257%；MoGe-2 900/60.099%；UniDepthV2 809/62.377%；VGGT1004/42.860%。完整事件/重力/相机/物理组与最大误差见bounded_depth.json。边界失败不能通过只报成功均值隐藏。我们此前相同共同帧未知速度centred仍2.909%均值、26.482%最大，未改变预测和默认。

先验速度、仰角及g范围已对齐，表示仍未完全对齐：抬升软L1尺度为1米而前向拟合为1像素；抬升中心抛物线没有我们同样的旋转几何和阻力表示。有限SLSQP初值不保证全局最优。VGGT缓存神经输入可能含额外帧，因此不能声称最终严格公平。无新GPU任务，所有本地实验已结束。下一步优先安排仅共同帧的VGGT推理：先核实NSCC空闲，确认仅在本人目录，重新生成共同帧任务与帧ID映射；不重复其他五种单帧深度推理。然后冻结对比协议，在未调优物理设置检验；最大误差目标仍未达到。

## VGGT共同输入完成（2026-10-09）

502条重推理完成，h100-1进程退出，GPU0已释放。与534条完全同输入缓存合并，核验1036唯一ID、模型名、完整有序帧号及深度长度；不按深度质量或目标误差决定复用。共同帧未知速度有界基线：981/1038成功（94.51%），54条边界、1条缺有效深度、2条无共同帧；成功均值46.881%、最大576.514%。旧缓存有界均值42.860%仅作为历史结果保留。完整事件/重力/相机/物理组统计vggt_common_result.json；并未改动我们的估计，也不能证明全局最优。

下一步：审计单帧深度输入及窗口协议剩余差异，冻结声明几何/速度边界/实际帧协议，再检验尚未调优物理设置；未知速度低帧率尾部仍需独立观测改进。不要重跑已完成VGGT任务。网络维护团队旧标识无效，新列表未找到同名任务，GPU现已释放，通知仍未声称送达。

## Observation capacity audit (2026-10-09)

All five single-image depth models match retained video/frame/box/intrinsics inputs on 1036/1036 records. Extra cached frames are independently inferred; numerical validity is still checked by the fitter. See single_depth_input_audit.json. VGGT alignment is already complete; representation and loss-unit differences remain.

At 15fps, 191/1038 declared windows cannot contain four cadence frames, two additional windows lack tracked frames, and 845 contain at least four. All failures remain in the denominator. sampling_audit15.json retains event/gravity/camera/physics-group statistics. No interpolation, refitting or GPU job was used.

Next: audit additional non-simultaneous free-flight windows of the same object, with independent velocities and shared gravity, providing identical added information to baselines. Freeze any rules before unseen physics-setting validation. Existing sufficient-frame tails still require observation improvements.

## 跨窗口开发控制（2026-10-09）

同一对象的非同时自由飞行窗口覆盖502/1038条，按真实15fps帧数最多、ID固定排序选择，未按g或误差选择窗口。191条采样容量不足中仅8条存在额外窗口；14条既有>10%尾部中仅4条有额外窗口。跨窗口声明位置、姿态、可见性仍来自仿真，不能声称video-only；新增信息应同步提供基线。覆盖及全部分组见cross_window_coverage15.json。

对4条可用尾部加8条固定哈希对照完成配对开发试验：各窗口速度独立、g共享，不提供参考速度；12/12拟合返回ok。原均值5.804%、最大14.428%；联合均值8.217%、最大60.905%。尾部4条均值11.884%→19.694%；哈希对照8条均值2.764%→2.478%。这是依赖已知误差的诊断，不是总体或独立测试；有限方向假设及优化器仍沿用实验joint_fit，不保证全局收敛。cross_window_pilot15.json保留完整逐条。没有按结果回退、删除难例，也没有改变默认。

3项相关测试通过，所有本地试验已退出，无GPU任务。跨窗口联合不能作为尾部修复，暂不扩大全量或重新跟踪。下一步检查联合60.9%案例的辅助窗口观测残差及声明初始姿态/自旋时刻一致性，先用相同时刻的投影真值框作诊断控制，区分新窗口模型误差与分割偏差；真值不得进入正式预测。之后才决定改模型或观测损失，保留失败方案和覆盖分母。

## Cross-flight oracle control (2026-10-09)

On the same previously selected 12-case pilot, projected boxes at identical times give joint maximum error 0.174108%, with all 12 ok. Independent-window maximum is 0.822801%. Declared initial position/rotation/angular velocity match compact states exactly. This diagnostic supports observation bias, not a formal improvement; no target gravity or reference velocity enters the fitter. Some windows retain up to two tiny/offscreen frames. See cross_window_diagnosis15.json. Default predictions remain unchanged; three relevant tests passed and local jobs ended.

## Auxiliary observation control (2026-10-09)

On the existing 12 selected development pairs, video-box quadratic smoothness does not reliably identify gravity bias: one auxiliary error is 21.68% with only 0.454px maximum smoothness RMS; another error is 0.023% with 5.245px RMS. Two T03 auxiliary errors exceed 180% without offscreen or declared hidden edges. A fixed independent-physics-residual RMS scale, floored at 1px, keeps every observation but gives 7.563% mean / 51.052% maximum, worse than original single-window 5.804% / 14.428%. All 12 return ok. These scales are not calibrated uncertainties. Default unchanged; see cross_window_edges15.json and cross_window_weighted15.json.

Auxiliary visual QA inspected nine frames from three already selected >60% auxiliary-error cases. Two T03 flights become border-clipped; T08 has a large tracking gap and top-border position disagreement, so identity drift cannot be excluded. Applying the existing fully-visible/unclipped four-real-frame admission rule to the 12 pilot auxiliary windows leaves 9 eligible, including the problematic T08 with four frames. This is an input audit only: all primary items remain, no refit or performance improvement is claimed. Dataset imagery stays local. See auxiliary_common_audit15.json and cross_window_visual15.json.

Fixed auxiliary admission pilot: all 12 primary items retained, 9 admitted auxiliaries and 3 primary-only. Mean/max 5.804%/14.428% -> 4.537%/13.606%, with 12/12 ok. No target-error fallback; auxiliary trimming preserves absolute motion time through t0 adjustment. Four relevant tests passed. This selected development pilot is not overall or unseen validation. The all-1038 control is running; only a complete unique-ID output may support whole-set conclusions.

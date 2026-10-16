# What The Video Shows

- **Human data.** 64 egocentric phone clips of one hand moving cards, a cube, a cup, a cylinder or a duster; 8 clips include a drop. No robot, no action labels.
- **Robot data.** 30 scripted Panda demonstrations in simulation: random start and goal positions, random objects (cube, cuboid, cylinder) and clutter. 9 of the 30 are deliberately imperfect (released mid-carry, edge grasps, missed grasps), so the model also sees unstable grasps.
- **Latent state.** Frozen DINOv3 patch tokens (14 × 24 tokens × 384) of six frames 0.2 s apart, shown with PCA colours.
- **Latent action.** A self-attention encoder reads each pair of consecutive frames and squeezes the change into a 2 × 8 bottleneck: one latent action per pair, five per window.
- **World model.** A causal latent action decoder predicts F̂₁ … F̂₅ from F₀ and z₁ … z₅. F̂ₖ attends to z₁ … zₖ only, so one forward pass predicts a second ahead for planning. Half of every training batch is human, half is robot; the only loss is MSE against the true tokens.
- **Inverse dynamics model.** Trained separately on 180 labelled play episodes to map two frames to a robot action. The world model never sees these labels; they only turn predicted frames into commands. A second IDM trained on Robotiq play data drives the Robotiq gripper with the same world model.
- **Planning.** CEM over latent actions scores predicted frames against a goal image; the IDM turns the best prediction into commands; replan every 0.2 s.
- **Results.** Fixed-seed reach (5) and carry (10) tasks per condition; success is the end effector within 5 cm of the goal pose. Co-trained + Panda 5/5 and 6/10, co-trained + Robotiq 3/5 and 5/10, human-only 1/5 and 2/10.

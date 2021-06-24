from datetime import datetime
import math
import os
import time

import pybullet as p

#clid = p.connect(p.SHARED_MEMORY)
import pybullet_data

# The path to the bullet 3 data folder, downloaded from:
# https://github.com/bulletphysics/bullet3/tree/master/data
# I'm not sure why, but that collection of data contains some files that are
# missing at the `pybullet_data.getDataPath()` path, such as cube.urdf.
BULLET_3_DATA = os.path.join(os.path.dirname(__file__), 'data')

p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setAdditionalSearchPath(BULLET_3_DATA)
p.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)
kuka_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(kuka_id, [0, 0, 0], [0, 0, 0, 1])
kuka_end_effector_index = 6
numJoints = p.getNumJoints(kuka_id)
if (numJoints != 7):
  exit()

time.sleep(5)
p.loadURDF('cube.urdf', [2, 2, 5])
p.loadURDF('cube.urdf', [-2, -2, 5])
p.loadURDF('cube.urdf', [2, -2, 5])

#lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
#joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i in range(numJoints):
  p.resetJointState(kuka_id, i, rp[i])

p.setGravity(0, 0, -10)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 0

count = 0
useOrientation = 1
useSimulation = 1
useRealTimeSimulation = 1
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 15

logId1 = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, "LOG0001.txt", [0, 1, 2])
logId2 = p.startStateLogging(p.STATE_LOGGING_CONTACT_POINTS, "LOG0002.txt", bodyUniqueIdA=2)

for i in range(5):
  print("Body %d's name is %s." % (i, p.getBodyInfo(i)[1]))

while 1:
  if (useRealTimeSimulation):
    dt = datetime.now()
    t = (dt.second / 60.) * 2. * math.pi
  else:
    t = t + 0.1

  if (useSimulation and useRealTimeSimulation == 0):
    p.stepSimulation()

  for i in range(1):
    pos = [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
    #end effector points down, not up (in case useOrientation==1)
    orn = p.getQuaternionFromEuler([0, -math.pi, 0])

    if (useNullSpace == 1):
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(kuka_id, kuka_end_effector_index, pos, orn, ll, ul,
                                                  jr, rp)
      else:
        jointPoses = p.calculateInverseKinematics(kuka_id,
                                                  kuka_end_effector_index,
                                                  pos,
                                                  lowerLimits=ll,
                                                  upperLimits=ul,
                                                  jointRanges=jr,
                                                  restPoses=rp)
    else:
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(kuka_id,
                                                  kuka_end_effector_index,
                                                  pos,
                                                  orn,
                                                  jointDamping=jd)
      else:
        jointPoses = p.calculateInverseKinematics(kuka_id, kuka_end_effector_index, pos)

    if (useSimulation):
      for i in range(numJoints):
        p.setJointMotorControl2(bodyIndex=kuka_id,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)
    else:
      #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
      for i in range(numJoints):
        p.resetJointState(kuka_id, i, jointPoses[i])

  ls = p.getLinkState(kuka_id, kuka_end_effector_index)
  if (hasPrevPose):
    p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
    p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
  prevPose = pos
  prevPose1 = ls[4]
  hasPrevPose = 1
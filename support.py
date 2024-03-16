from utils import *
from searchClasses import *
import math

"""
this file contains class definitions and calculation functions for the motion planning problems
该文件包含 运动规划问题的类定义和计算函数
Contains general tools for distance calculation and plotting
包含用于 距离计算和绘图的通用工具
为不同系统提供转向功能

导入了3个文件：utils.py，searchClasses.py，environment.py
"""

def withinBounds(steered_node,bounds):
    return (bounds[0] < steered_node[0] < bounds[2]) and (bounds[1] < steered_node[1] < bounds[3])

def  getRandomNode(bounds, iteration, goal_bias, bias_rate, end_region, informed=False, ellipse=None):
    """
    Purpose:    samples random node from within given bounds
                给定边界内随机采样
    :params:    ellipse is an Ellipse object
                ellipse是椭圆对象
    """

    if not (iteration % bias_rate) and (not iteration == 0) and (goal_bias):
        return end_region.centroid.coords[0]   # 一定概率采样终点区域 GOAL_BIAS=true
    else:
      if not ellipse is None and informed:  # ellipse=None 可能是informed rrt*
        return ellipse.randomPoint()
      else:
        rand_x = random.uniform(bounds[0], bounds[2])    # xmin和xmax随机取值
        rand_y = random.uniform(bounds[1], bounds[3])    # ymin和ymax随机取值
        return (rand_x, rand_y)

def steerPath(firstNode, nextNode, dist, eta=0.5):
    """
    Purpose:    A super naive way of steering towards a sampled point
                用于简单的向采样点转向

    :params:    tuples as input, dist between them gives is distance to move 组元作为输入，它们之间的距离就是移动的距离
                eta is a float bounding:    eta 是一个浮点边界
                eta <= epsilon, where |x1-x2| < eta for all steering |x1 - steer(x1,x2)| < epsilon
                eta <= epsilon，其中|x1-x2| < eta 适用于所有转向|x1-steer(x1,x2)|<epsilon
    """
    # 用于直接向采样点转向

    # avoiding errors when sampling node is too close   避免采样节点太近时出现误差
    if dist == 0:
        dist = 100000   # 避免出现除0

    hori_dist = nextNode[0] - firstNode[0]  # signed
    vert_dist = nextNode[1] - firstNode[1]  # signed
    dist = dist/2  # originally used to divide hori_dist and vert_dist on   原本用于在"hori_dist"和"vert_dist"上划分

    if True:
        if math.sqrt(hori_dist**2 + vert_dist**2) > eta:   # 考虑了距离限制，确保转向距离不能超过步长
            angle = np.arctan2(vert_dist, hori_dist)
            hori_dist = eta * np.cos(angle)
            vert_dist = eta * np.sin(angle)

        return (firstNode[0] + hori_dist, firstNode[1] + vert_dist)     # 向node_rand方向前进eta步返回steered_node

    else:   # 未被使用
        return (firstNode[0] + hori_dist/dist, firstNode[1] + vert_dist/dist)

def constrainedSteeringAngle(angle, max_angle=3.14/10):
    """
    Keeps steering angle within a certain max  将转向角保持在一定的最大范围内

    :param:     angle (float): angle to constrain
                max _angle(float): maximum steering angle
    """
    if angle > max_angle:
        return max_angle
    elif angle < -max_angle:
        return -max_angle

    return angle

def calculateSteeringAngle(firstNode, theta, nextNode, L, maxx=np.pi / 10.):
    # 考虑目标点与前轮之间的相对角度，计算自行车的转向角度，并确保其在一定范围内

    fWheel=(firstNode[0]+L*np.cos(theta), firstNode[1]+L*np.sin(theta))  # 计算前轮位置
    # arctan2 gives both positive and negative angle -> map to 0,2pi   arctan2可以给出正角和负角 -> 映射到 0,2pi
    angle = np.mod(np.arctan2(nextNode[1]-fWheel[1], nextNode[0]-fWheel[0]), 2*np.pi)
    rel_angle = angle - theta
    # enforce steering to within range RELATIVE to theta -> map to (-pi,pi)  将转向控制在与theta相对的范围内 -> 映射到(-pi,pi)
    rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle))    # 确保在(-pi,pi) 对转向角度进行限制

    return constrainedSteeringAngle(rel_angle, maxx)   # max_steer is default


def bicycleKinematics(theta, v, L, delta, dt):

    dth  = v/L*np.tan(delta)    # 使用运动学模型计算 转向角速度增量 和x,y增量
    dx   = v * np.cos(theta)
    dy   = v * np.sin(theta)
    return (dx, dy, dth)

def bicycleKinematicsRK4(initState, theta, v, L, delta):
    x0, y0 = initState

    k1t = v/L*np.tan(delta)
    k1x = v * np.cos(theta)
    k1y = v * np.sin(theta)

    k2x = v * np.cos(theta + k1t/2)
    k2y = v * np.sin(theta + k1t/2)

    k3x = v * np.cos(theta + k1t/2)
    k3y = v * np.sin(theta + k1t/2)

    k4x = v * np.cos(theta + k1t)
    k4y = v * np.sin(theta + k1t)

    dx = (k1x + 2*(k2x + k3x) + k4x) / 6
    dy = (k1y + 2*(k2y + k3y) + k4y) / 6
    dth = k1t  # has no effect throughout; kjt is equal to k1t for all j    自始至终没有影响；对于所有j,kjt=k1t

    return dx, dy, dth

def steerBicycleWithKinematics(firstNode, theta, nextNode, dt):
    """
    Purpose:  Steer the kinematic bicycle model from firstNode towards nextNode
              将运动学自行车模型从firstNode转向nextNode

    If delta could be stored, a more realistic implementation would be to constrain delta considering the steering angle already made!
    如果可以存储 delta，那么更现实的实施方法是在考虑到已经做出的转向角的情况下对 delta 进行约束！
    :params:    firstNode and nextNode is a tuple of x,y pos of back wheel
                firstNode and nextNode是当前点和目标点的后轮位置(x, y)坐标
                theta is a float of bicycle frame angle wrt. x axis

                theta表示自行车车架相对于x轴的角度

    """
    # 根据当前位置和朝向，将自行车模型从一个点转向另一个点

    L, v = (0.25, 0.5)  # frame length L, velocity v    自行车的车架长度L和速度v

    # 1) Find the new delta parameter for steering wheel  找到新delta参数
    delta = calculateSteeringAngle(firstNode, theta, nextNode, L)   # 计算转向角度，是将自行车转向下一个点所需的角度

    # 2) Use that as input to kinematics   将其作为运动学的输入
    # Old Euler Method vs more precice Runge-Kutta fourth order   欧拉法与更精确的Runge-Kutta四阶法对比
    # dx, dy, dth = bicycleKinematics(theta, v, L, delta, dt)   # 实现了基于运动学的自行车模型的运动学方程
    # delta是转向角度
    dx, dy, dth = bicycleKinematicsRK4(firstNode, theta, v, L, delta)   # 使用Runge-Kutta四阶方法来计算自行车模型的运动学
    # dx,dy是速度增量 ,dth是角速度增量

    # 计算新位置(xnew, ynew)
    thetanew    = theta        + dth * dt
    xnew        = firstNode[0] + dx * dt
    ynew        = firstNode[1] + dy * dt

    # TODO: limit distance travelled to eta
    # 限制行驶距离到eta，没有实现

    if False:   # 为啥不使用？
        if math.sqrt(hori_dist**2 + vert_dist**2) > eta:
            angle = np.arctan2(vert_dist, hori_dist)
            hori_dist = eta * np.cos(angle)
            vert_dist = eta * np.sin(angle)
            # 调整水平距离和垂直距离，使其模等于eta，但方向保持不变
        return (firstNode[0] + hori_dist, firstNode[1] + vert_dist)

    # map thetanew to (0,2pi)
    return (xnew, ynew), np.mod(thetanew, 2 * np.pi)

def steerBicycleWithDynamics(firstNode, theta, nextNode, dt, velocity):
    """
    Purpose:  Incorporates dynamic constraints on simple bicycle model  在简单的自行车模型中加入 动态约束条件
              Note that (x,y)-position is now calculated from CG instead of back wheel as in the kinematic version
              请注意(x,y)位置现在是从CG(质心)开始计算,而不是像运动学版本那样从后轮开始计算
              Model used is from [Pepy, Lambert, Mounier: 2006]
              center of gravity

        :params:
                firstNode (tuple(floats)): x,y pos of back wheel before  后轮前的x、y位置
                nextNode  (tuple(floats)): x,y pos of back wheel next    后轮后的x、y位置
                theta (float): bicycle frame angle wrt. x axis (0,2pi)   theta：自行车框架相对于x轴的角度（0,2pi）
                dt (float):

    General note: ddx and ddelta would be inputs    ddx 和 ddelta 将是输入端
    """
    L, vx = (0.25, 0.5)  # frame length L, velocity v
    Lr = Lf = L/2  # setting distance from CG to front/back wheel to be equal 设置CG到前后轮的距离相等

    # Standard bike dimensions
    Lstd = 1
    mstd = 10
    Istd = (1**2 + 0.2**2) * mstd / 12  # 自行车的转动惯量
    tire_radius_std = 0.33 * Lstd

    # Scale to wanted length (course estimation to fit the test environment)   按所需长度缩放（根据测试环境估算航程）
    scale = L / Lstd
    m = scale * mstd
    I = scale * Istd
    tire_radius = scale * tire_radius_std

    delta = calculateSteeringAngle(firstNode, theta, nextNode, L, maxx=np.pi/20)

    # vx is velocity in direction of theta  vx 是θ 方向上的速度
    # vy is lateral velocity of cg (can move sideways if vx > 0 and delta != 0)
    # vy 是cg的横向速度（如果 vx > 0 且 delta != 0，则可横向移动）
    # dth is rate of change of orientation vs x-axis   dth 是方向相对于 x 轴的变化率
    # velocity at firstNode
    (vx, vy, dth) = velocity
    r = dth

    # values used from
    Cf = Cr = 1  # usually quite large, but mass is going to be very small

    a = b = tire_radius
    cf = (vy + Lr * a)
    # cf = np.arctan2(np.sin(cf), np.cos(cf))
    cr = (vy - Lr * b)
    # cr = np.arctan2(np.sin(cr), np.cos(cr))

    alphaf = np.arctan(cf/vx) - delta  # before: this entire set was in arctan
    alphar = np.arctan(cr/vx)

    # print(alphar, alphaf)
    # adding minus sign infront of forces
    Ff = -(Cf) * alphaf
    Fr = -(Cr) * alphar

    # euler integration
    dvx = 0  # just for clarity

    # dr = Lr * (Ff * np.cos(delta) - Fr) / I
    dr = (m*a*np.tan(delta)*(dvx-r*vy) + a * Ff / np.cos(delta) - b * Fr ) / I

    # dvy = (Ff * np.cos(delta) + Fr) / m - vx * r
    dvy = np.tan(delta)*(dvx-r*vy) + (Ff/np.cos(delta) + Fr)/m - vx*r

    rnew = dth + dr * dt
    thetanew = theta + rnew * dt
    dx = vx * np.cos(thetanew) - vy * np.sin(thetanew)
    dy = vx * np.sin(thetanew) + vy * np.cos(thetanew)

    vynew = vy + dvy * dt
    vxnew = vx + dvx * dt
    xnew = firstNode[0] + dx * dt
    ynew = firstNode[1] + dy * dt

    newstate = (xnew, ynew)

    # map thetanew to (0,2pi)
    return newstate, np.mod(thetanew, 2 * np.pi), (vxnew, vynew, rnew)

def steeringFunction(steer_f, node_nearest, node_rand, node_dist, dt, eta=0.5):
    steered_velocity = (0.5, 0.0, 0.0)
    steered_theta = 0
    # dt是时间步长 eta是距离阈值  node_nearest的最近节点的状态信息，包括位置、角度和速度   node_dist两点距离

    if steer_f == None:     # none  不执行转向操作
      steered_node = steerPath(node_nearest.state, node_rand, node_dist, eta)

    if steer_f == False:    # kinematic model
      steered_node, steered_theta = steerBicycleWithKinematics(node_nearest.state, node_nearest.theta, node_rand, dt)
      steered_velocity = (0.5, 0.0, 0.0)    # 速度固定  steered_node是转向后的节点也就是new节点

    if steer_f == True:     # dynamic model
      steered_node, steered_theta, steered_velocity = steerBicycleWithDynamics(node_nearest.state, node_nearest.theta, node_rand, dt, velocity=node_nearest.velocity)

    return steered_node, steered_theta, steered_velocity

def sampleQuadcopterInputs(searchNode, nextnode, L, m, I, g, u):  # 计算四旋翼的输入（没有用到）
    """
    Notes:  Differ between points that are above and below?
            This has to depend on the previous inputs and velocities

            Needs to have the ability to reduce both gains to decend
            - yddot will always be maintained if (u1+u2)*cos(th) == mg,
              so this is the parameters to adjust according to wanted altitude

            Make sure that th stays pretty small, indep of vertical movement
            - Extra: so that the dynamics might be linearizable?
    """

    pos = searchNode.state
    theta = searchNode.theta
    dx = nextNode[0]-pos[0]
    dy = nextNode[1]-pos[1]
    # arctan2 gives both positive and negative angle -> map to 0,2pi
    angle = np.mod(np.arctan2(dy, dx), 2*np.pi)

    # This is the angle between the CG and the next point. Remember that setting theta towards this angle makes the quadcopter take of +90 deg in relation to that! -> Find the relative angle between the direction the quadcopter should point towards (which is +90 deg of theta)
    rel_angle = angle - (theta + np.pi)

    # Map to (-pi,pi). This range is directly applicable to theta and the change we need to enforce on it
    rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle))

    # Now to how we change u1,u2. Remember that u1 is the rightmost input, so if it is larger than u2, theta increases and the quadcopter tips leftward.
    #current_accel = ( (u1+u2)*np.sin(theta)/m, (u1+u2)*np.cos(theta)/m - g )

    # first, rotate the drone according to rel_angle
    # adjust in such a way that yddot = 0 initially?

    # find the magnitude of this angle to decide when to do nothing, or just be happy with a drone that flies through the end region
    # then, find out if the magnitude of the vertical thrust needs to be increased, reduced (or unchanged if yddot wasn't enforced earlier)

    ### insert code for finding u1,u2 here

    # VOILA, we should have our new u1,u2's. As seen from differential flatness, they should be very similar, always

    ## Increase and decrease u1,u2 in relation to previous ones and some kind of maximum rate of change (not realistic to suddenly change u1,u2 to be totally previous)

    return (u1, u2)

def steerWithQuadcopterDynamics(searchNode, nextNode, dt):
    """
    searchNode  is supposed to be nodeNearest as SearchNode object

    """
    # TODO: Make a function that samples u1,u2 relative to what searchNode had before

    L = 0.25     # length of rotor arm
    m = 0.486    # mass of quadrotor
    I = 0.00383  # moment of inertia
    g = 9.81     # gravity

    x, y = searchNode.state
    th  = searchNode.theta
    xdot, ydot, thdot = searchNode.velocity
    u1, u2 = searchNode.u

    xddot = -1/m * np.sin(theta) * (u1 + u2)
    yddot =  1/m * np.cos(theta) * (u1 + u2)
    thddot =  L/I * (u1 - u2)

    xdotnew = xdot + xddot * dt
    ydotnew = ydot + yddot * dt
    thdotnew = thdot + thddot * dt

    xnew = x + xdotnew * dt
    ynew = y + ydotnew * dt
    thnew = th + thdotnew * dt

    return (xnew, ynew), thnew, (u1, u2)

def obstacleIsInPath(firstNode, nextNode, env, radius):
    """
    Purpose:  Checks for an obstacle in line between firstNode and nextNode
    检查firstNode和nextNode之间的直线上是否有障碍物
    :returns: A boolean for collision or not    表示是否碰撞的布尔值
    """

    # Point from shapely
    start_pose = Point(firstNode).buffer(radius, resolution=3)
    end_pose = Point(nextNode).buffer(radius, resolution=3)

    # LineString from Shapely
    line = LineString([firstNode, nextNode])
    expanded_line = line.buffer(radius, resolution=3)

    if env.obstacles is not None:
        for i, obs in enumerate(env.obstacles):
            # Check collisions between the expanded line and each obstacle
            if (expanded_line.intersects(obs)):
                return True

    return False

def goalReached(node, radius, end_region):
    """
    :params:    node is a tuple (xpos, ypos)
                radius is a scalar - size of the robot 机身的大小
                end_region is a polygon of four tuples drawn clockwise: lower left, upper left, upper right, lower right
                目标区域是多边形
    """
    # returns a boolean for node tuple + radius inside the region
    # 返回区域内节点元组 + 半径的布尔值
    return end_region.contains(Point(node))     # Polygon的方法用法：判断某个点是否在某多边形内

def getMaximumNeighborRadius(ETA, no_nodes):
  """
  Purpose:  Implements the maximum radius from [Karaman and Frazzoli, 2013]
            Note that (log(x) / x)**(0.5) is always lower than 0.61
            注意 (log(x) / x)**(0.5) 总是低于 0.61
            论文中的公式
  """
  return ETA * math.sqrt(math.log(no_nodes) / no_nodes)

def nearestNeighbors(s_f, graph, newNode, k, ETA, no_nodes, radius=1):
    """
    Purpose:  find tuple in graph that is closest to newNode
              在图中找出最接近newNode的元组

    :params:  s_f(没有使用) is steering function choice TODO: how to use it?
              graph is Graph object
              newNode is a SearchNode(node_steered)
    """

    # 根据需要查找的数量获取所有近邻，并使用Karaman,2013,RRTstar中计算效率较低的方法，通过最大化半径在球内寻找非整体约束，从而降低计算量
    # Get all nearest neighbors according to number to look for, and use the computationally inefficient way of keeping calculations down from Karaman, 2013, RRT star for nonholonomic constraints by maximuzing radii to look within a ball
    # 利用最近邻算法找到了与目标状态最接近的状态集合
    states = []
    it = 0
    loc_pos = 0
    for node in graph._nodes:
        states.append([node.state[0], node.state[1]])   # 提取位置信息
        if node.state == newNode.state:  # 记录newNode位置信息在[]中的位置
            loc_pos = it    # 在[]中第一位
        it += 1

    X = np.array(states)
    nbrs = NearestNeighbors(n_neighbors=k, radius=radius, algorithm='ball_tree').fit(X)  # 近邻搜索算法,用balltree算法

    distances, indices = nbrs.kneighbors(X)     # 返回每个状态的最近邻的距离和索引

    SN_list = [graph._nodes[ind] for ind in indices[loc_pos]]

    return SN_list

def minCostPath(s_f, k, SN_list, node_min, node_steered, env, r, max_rad=1):
  """
  Purpose:  Finds the node that contributes to the cheapest path out of the k-1 nearest neighbors
            从 k-1个最近的邻居中找出为最近路径做出贡献的节点
  """
  if s_f == None:   # rrt*
    # For all neighbors, add the steered node at the spot where it contributes to the lowest cost. Start counting at 1 since pos 0 is the node itself
    # 对于所有邻居，将被引导的节点添加到成本最低的位置。从1开始计数，因为位置0就是节点本身
    for j in range(1, k):
      node_near = SN_list[j]
      if not obstacleIsInPath(node_near.state, node_steered.state, env, r):     # 检测两点连线是否有障碍物
        cost_near = node_near.cost+eucl_dist(node_near.state, node_steered.state)
        if cost_near < node_steered.cost:
          node_min = node_near

    # Update parent and cost accordingly
    # 更新父节点和相应的cost
    node_steered.parent = node_min
    relative_distance = eucl_dist(node_min.state, node_steered.state)
    newcost = node_min.cost + relative_distance
    node_steered.cost = newcost
    return node_min, relative_distance

  elif s_f == False:    # 运动学rrt*
    anyReachable = False

    for j in range(1, k):
      node_near = SN_list[j]
      if not obstacleIsInPath(node_near.state, node_steered.state, env, r) and nodeIsReachable(node_near, node_steered):

        anyReachable = True

        cost_near = node_near.cost+kin_dist(node_near, node_steered)
        if cost_near < node_steered.cost:
          node_min = node_near

    node_steered.parent = node_min
    relative_distance = kin_dist(node_min, node_steered)
    newcost = node_min.cost + relative_distance
    node_steered.cost = newcost

    return node_min, relative_distance, anyReachable

def nodeIsReachable(node_steered, node_near, max_steer=math.pi/10):   # 通过运动学约束判断
  """
  Purpose:  Tests to see if node_near is within the range of node_steered
            测试 node_near 是否在 node_steered 的范围内
  """
  x1, y1 = node_steered.state
  x2, y2 = node_near.state
  dx = x2 - x1
  dy = y2 - y1
  rel_angle = np.arctan2(dy, dx)    # between -pi and pi
  if rel_angle > max_steer or rel_angle < - max_steer:
    return False

  return True

def rewire(ax, bounds, s_f, graph, node_min, node_steered, env, radius, SN_list, k, max_steer):
  """
  Purpose:  Rewires graph to remove sub-optimal cost paths for k nearest neighbors
            重新布线图形，删除 k 个近邻的次优成本路径
  """

  # SN_list gives all neighbors within max radius. Manipulate the list to only contain reachable nodes according to current body frame angle and max steering angle
  # SN_list列出了最大半径范围内的所有邻居。根据当前车身框架角度和最大转向角，对列表进行处理，使其只包含可到达的节点

  for j in range(1, k):
    node_near = SN_list[j]  # gives all neighbors within max radius 给出最大半径内的所有邻居节点
    if True or node_near is not node_min:

      if s_f == False:  # kinematic behavior
        if not nodeIsReachable(node_steered, node_near, max_steer):
          continue

      if not obstacleIsInPath(node_near.state, node_steered.state, env, radius):

        if s_f == False:  # kinematic behavior
          newcost = node_steered.cost+kin_dist(node_steered, node_near)
        else:  # holonomic behavior
          newcost = node_steered.cost+eucl_dist(node_steered.state, node_near.state)

        if newcost < node_near.cost:
          node_parent = node_near.parent
          graph.remove_edge(node_parent, node_near)
          node_near.parent = node_steered
          node_near.cost = newcost

          if s_f == False:
            dist = kin_dist(node_steered, node_near)
          else:
            dist = eucl_dist(node_steered.state, node_near.state)

          graph.add_edge(node_steered, node_near, dist)
          graph.updateEdges(node_near)
          # the node cost has changed 节点成本发生变化
import time
from support import *

"""
imports utils and searchClasses as well
同时导入了四个文件文件：support.py，utils.py，searchClasses.py，environment.py
"""

"""
Globals
"""

# Plotting params   画图参数
plotAll = False     # 绘制最佳路径和信息椭圆
realTimePlotting = False    # 实时绘图
stepThrough = False     # 没用到
onePath = True  # 一条路径
restartRRT = False
plotAllFeasible = False     # # 没用到
noFeas = 100
pauseDuration = 0.001   # 没用到
rewireDuration = 0.001  # 没用到

# Config params    配置参数
MAX_ITER = 10000
RRTSTAR = True
INFORMED = True    # informed rrt*
PATH_BIAS = False   # 没用到
PATH_BIAS_RATE = MAX_ITER   # Not used for now  没用到
STEERING_FUNCTION = False    # None is none（全向）,False（加载自行车运动学模型）,True（加载自行车动力学模型）动力学不起作用
GOAL_BIAS = True    # 没用到
GOAL_BIAS_RATE = 30     # 采样率
MIN_NEIGH = 3   # 没用到
MAX_NEIGH = 50  # 设定最大邻节点数量
MAX_STEER = math.pi / 10
ETA = 0.5     # 0.5 standard, only relevant for holonomic  与完整性约束有关
DT = 0.1    # steer的时候步长

"""
RRT算法
"""

def rrt(ax, bounds, env, start_pos, radius, end_region, start_theta):
    """
    Adding tuples nodes_list -> represent ALL nodes expanded by tree
    添加元组nodes_list->表示按树展开的所有节点
    """

    nodes = [start_pos]   # 初始化一个节点列表，将start加入nodes中
    graph = Graph()   # 创建一个图对象，有节点，边，标签函数，节点位置属性 给定的图数据结构
    start_node = SearchNode(start_pos, theta=start_theta)   # 创建搜索节点对象  theta=3*pi/4
    graph.add_node(start_node)
    goalPath = Path(start_node)
    bestPath = goalPath   # path object 路径目标
    bestCost = 100000
    feasible_paths = []
    goalNodes = []
    info_region = None
    sampling_rate = GOAL_BIAS_RATE  # 采样率

    printRRTstar() if RRTSTAR else printRRT()   # 条件表达式
    print("Looking for path through", MAX_ITER, "iterations")
    for i in range(MAX_ITER):  # random high number
        EprintProgressBar(i, MAX_ITER, "Progress", suffix=("/ iteration "+str(i)))  # 打印迭代进度
        
        node_rand = getRandomNode(bounds, i, GOAL_BIAS, GOAL_BIAS_RATE, end_region, INFORMED, info_region)  # 据传入的参数返回一个随机节点

        node_nearest, node_dist = nearestEuclSNode(graph, node_rand)  # 查找距离node_rand最近的节点

        steered_state, steered_theta, steered_velocity = steeringFunction(STEERING_FUNCTION, node_nearest, node_rand, node_dist, DT, ETA)   # 转向函数

        if not withinBounds(steered_state, bounds):  # 确保生成的节点在规定的运动空间bounds内
          continue
      
        # update dist after steering    更新搜索节点的状态信息
        node_dist = eucl_dist(node_nearest.state, steered_state)
        node_steered = SearchNode(steered_state, node_nearest, node_nearest.cost + node_dist, theta=steered_theta, velocity=steered_velocity)

        if node_steered.state not in nodes:
            if not obstacleIsInPath(node_nearest.state, node_steered.state, env, radius):   # 连线是否有障碍物

                nodes.append(node_steered.state)
                graph.add_node(node_steered)
                node_min = node_nearest     # 最近节点
                no_nodes = len(graph._nodes)    # 图中总节点数

                if RRTSTAR:
                  # limit nearest neighbor search to be 10% of the entire graph 限制最近邻节点数量为整个图节点的10%
                  k = no_nodes-1 if no_nodes < MAX_NEIGH else int(no_nodes*0.1)
                  if (no_nodes > k and k > 0):
                    # 查找通过最短的无碰撞路径连接到node_steered的节点
                    # Find node that connects to node_steered through the cheapest collision-free path
                    max_n_radius = getMaximumNeighborRadius(ETA, no_nodes)  # 最大领域半径，限制最近邻搜索的范围

                    SN_list = nearestNeighbors(STEERING_FUNCTION, graph, node_steered, k, ETA, no_nodes, max_n_radius)  # 找到距离给定节点node_steered最近的一组邻居节点列表

                    node_min, rel_dist, ok = minCostPath(STEERING_FUNCTION, k, SN_list, node_min, node_steered, env, radius)    # 最小成本路径计算

                    if not ok:  # no reachable nodes  没有找到节点 ok：是否找到可行的路径
                      continue

                    graph.add_edge(node_min, node_steered, rel_dist)

                    # Rewiring the remaining neighbors
                    # 为其余邻居重新布线
                    rewire(ax, bounds, STEERING_FUNCTION, graph, node_min, node_steered, env, radius, SN_list, k, MAX_STEER)

                    if False and plotAll and not i%500:  # 在特定条件下实时绘制图形
                      plotEdges(ax, graph)
                      if realTimePlotting:
                        plt.draw()
                        plt.pause(0.01)

                  else:
                    # Not enough points to begin nearest neighbor yet
                    # 还没有足够的节点执行最近邻搜索
                    graph.add_edge(node_nearest, node_steered, node_dist)
       
                                      
                else:   # 有rrt*不执行
                  # No RRT Star - Don't consider nearest or rewiring
                  # 无rrt*  不考虑最近的或重新布线
                  graph.add_edge(node_nearest, node_steered, node_dist)

                  if plotAll and realTimePlotting:

                    plot_poly(ax, Point(node_steered.state).buffer(radius/3, resolution=5), color='blue', alpha=.6)

                    plt.draw()
                    plt.pause(0.01)
                    input("Enter to continue")

                    plot_poly(ax, Point(node_min.state).buffer(radius/3, resolution=5), color="red", alpha=.8)
                    plotNodes(ax, graph)
                    plt.draw()
                    plt.pause(0.01)
                    input("Enter to continue")

            else:
                # Avoid goal check if collision is found
                # 如果发生碰撞，则避免目标检查
                continue

        else: 
          # The node has already been sampled
          # 节点已采样
          continue

        # Check last addition for goal state
        # 检查目标状态的最后一个附加项  处理到达目标状态的逻辑
        if goalReached(node_steered.state, radius, end_region):     # 判断是否到达目标状态
            goalPath = Path(node_steered)
            goalNodes.append(node_steered)
            
            if onePath:     # 检查是否只需要找到一条路径  在满足特定条件时绘制一个椭圆
              if INFORMED and plotAll:  # informed rrt*
                # just to see what the ellipse is doing - testing plot etc
                # 只是想看看椭圆在做什么 - 测试绘图等
                info_region = Ellipse()
                info_region.generateEllipseParams(goalPath.path)
                info_region.plot(ax)

              return goalPath

            else:  # we allow looking for more paths 我们允许寻找更多的路径

              if not RRTSTAR:
                if len(feasible_paths) > noFeas:
                    break

              # Important that there is a new init of Path object each time
              # 重要的是，路径对象每次都有一个新的初始值
              feasible_paths = [Path(node) for node in goalNodes]
              costs = [pathObj.cost for pathObj in feasible_paths]
              idx = costs.index(min(costs))
              bestPath = feasible_paths[idx]

              if min(costs) < bestCost:
                bestCost = min(costs)
                bestNode = goalNodes[idx]

                if INFORMED:    # 启用了informed rrt*，则会相应地调整采样率
                  info_region = Ellipse()
                  info_region.generateEllipseParams(bestPath.path)
                  sampling_rate = 1.1 * sampling_rate  # bias less and less 偏差越来越少

                if plotAll:     # 绘制最佳路径和信息椭圆
                  plotListOfTuples(ax, bestPath.path)

                  if INFORMED:
                    # note that the end plot might show the goal path outside of the ellipse. This is just because of the rewiring
                    # 注意，终点图可能会显示椭圆外的目标路径，这只是因为重新布线
                    info_region.plot(ax)

                  if realTimePlotting:
                    plt.draw()
                    plt.pause(0.01)

                print("\nNew best cost: %0.3f" % bestCost + " ")

              if not RRTSTAR and not onePath and restartRRT:
                # 如果不是 RRT* 算法，且需要找到多条路径，并且启用了重新启动 RRT，则会重新启动搜索并继续搜索
                # Restart search and continue as long as the iterations allows
                # 只要迭代次数允许，就重新开始搜索并继续搜索
                nodes = [start_pos]
                graph = Graph()
                graph.add_node(SearchNode(start_pos, theta=start_theta))
                goalPath = Path(SearchNode(start_pos, theta=start_theta))

    if goalNodes != []:  # 用于从目标节点列表中找到具有最低成本的路径
      feasible_paths = [Path(node) for node in goalNodes]
      costs = [pathObj.cost for pathObj in feasible_paths]
      idx = costs.index(min(costs))
      bestPath = feasible_paths[idx]    # 找到成本最低的路径

    if False and plotAll and not restartRRT:
      plotEdges(ax, graph)

    return bestPath


"""
Tests
测试程序入口
"""

plots = True
if(plots):
    radius = 0.3
    xmin = -2; ymin = -3; xmax = 12; ymax = 8
    bounds = (xmin, ymin, xmax, ymax)

    goal_region = Polygon([(11, 7), (11, 8), (12, 8), (12, 7)])       # 顺时针方向   Polygon是表示多边形的库，设置目标区域并创建对象

    # cost of 16.35 is perfect lines  cost为16.35是最佳的线条
    environment = Environment('env_superbug.yaml')
    start = (0, 0)
    s_th = 3*np.pi/4   # 起始节点角度好像没用，每次都在变化？
    # 初始化环境对象：障碍物，边界，起始位置，起始节点角度

    # environment = Environment('env_slit.yaml');   start=(-1,1); s_th=0
    # environment = Environment('env_empty.yaml');  start=(-1,1); s_th=0

    # 打印看看结果
    print("environment=", environment.obstacles)
    print("environment.bounds=", environment.bounds)
    print("bounds=", bounds)

    # 画障碍物图
    ax = plot_environment(environment, bounds)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    st = time.time()
    goalPath = rrt(ax, bounds, environment, start, radius, goal_region, start_theta=s_th)     # ax是轴对象

"""
下面是matplotlib仿真画图
可视化环境以及路径规划的结果
"""

print(("\nTook %0.3f" % (time.time()-st)) + "seconds. \nPlotting...")
plot_poly(ax, Point(start).buffer(radius, resolution=5), 'blue', alpha=.3)
plot_poly(ax, goal_region, 'red', alpha=0.3)

plotListOfTuples(ax, goalPath.path, width=1.5)    # 用于在图形对象ax上绘制一组元组组成的路径
# plot_Bspline_From_List(ax,goalPath.path,bounds,sn=200) # for faster plotting 以加快绘图速度 b样条？

time_used = DT * len(goalPath.path)
ax.set_title("Best cost, time: %0.3f units in %0.1f s" % (goalPath.cost, time_used) )

q = input("Goal path should be found with cost " + str(goalPath.cost) +"\n Enter 'q' for anything else then plotting goalpath")

plt.close() if q == 'q' else plt.show()
plt.close()
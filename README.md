## Implementation of (Informed) RRT*

Scripts:
 - RRT.py (RRT, RRT* and Informed RRT*. Can be used with a holonomic system, simple bicycle model with kinematic constraints, and a more complex model with dynamic constraints)

 - SimplePendulum.py (implemented simple RRT for pendulum specifically. This was the inital version.)

 - DoubleIntegrator2D.py (stopped working on May 16., will not be prioritized to fix)

 - 

**脚本:**

- **RRT.py**（**RRT**、**RRT*** 和 **Informed RRT***，可与**完整系统**、**具有运动学约束的简单自行车模型**以及**具有动态约束的更复杂模型**一起使用）
- **SimplePendulum.py**（专门为**钟摆**实现了简单的 RRT，这是**初始版本**）**能运行**
- **DoubleIntegrator2D.py**（5月16日停止工作，不会优先修复，文件有问题）**舍弃**

Other:
- All files imports support.py, which implements steering functions for the different systems.
- support.py imports utils and searchClasses, which consists of implementation of plotting tools and classes used for the RRT algorithm
- An example of differential flatness is added, but will not run in this folder due to a lot of reasons, first and foremost since the support files made for it is in another directory, but also since the user don't have PYTHONPATH set correctly. Added due to display of an example discussed in the final report.

**其他：**

- 所有文件都导入**support.py**，它为**不同系统实现转向功能**
- **support.py** 导入 **utils** 和 **searchClasses**，其中包含**用于 RRT 算法的绘图工具和类的实现**
- 添加了**差分平坦度**的**示例**，但由于多种原因**不会在此文件夹中运行**，首先是因为**为其制作的支持文件位于另一个目录中**，而且还因为用户没**有正确设置 PYTHONPATH**,**由于显示最终报告中讨论的示例而添加**，运行不了

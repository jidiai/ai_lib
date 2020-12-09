## 平台地址
[ai_olympics](http://39.102.68.22/)

## 项目依赖

- `Pygame`
  - 版本至少为2.0, 使用命令`pip install pygame==2.0.0dev8`进行安装
- `Python 3.6`
- `Torch 1.7.0` 可选
  - 支持提交Torch训练后的模型.pth附属文件

## 目录结构

```
|-- platform_lib
	|-- README.md
	|-- run.py		// 本地调试运行环境
	|-- examples	// 提交运行文件示例	需包含 my_controller 函数输出policy
	    |-- randomagent.py  // 随机策略
	|-- simulators		// 模拟器
	|	|-- game.py
	|	|-- gridgame.py // 网格类模拟器接口
	|-- obs_interfaces		// observation 观测类接口
	|	|-- observation.py		// 目前支持Grid
	|-- env		// 游戏环境 
	|	|-- config.ini		// 相关配置文件
	|	|-- gobang.py
	|	|-- reversi.py
	|	|-- sokoban.py
	|	|-- chooseenv.py 
	|	|-- snakes.py
```

## 平台提交说明
1. 填写算法名称或描述，选择提交环境
2. 上传运行文件。运行文件需包含`my_controller` 函数的一个`submission.py`文件。
3. 上传附属文件（可选）。附属文件支持`.pth` `.py`类型文件。大小不超过100M，个数不超过5个。 

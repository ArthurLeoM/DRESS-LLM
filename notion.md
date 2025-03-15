代码rerun总结

- 创建conda环境
    - 注意environment.yaml中baukit改为
            - baukit @ git+https://github.com/davidbau/baukit.git@main
- 下载模型到本地
- 机器建议选用A100
    - 依赖的版本比较老，不支持过新的版本
    - 14B模型，运行时最高显存占用32G
- get_activation
    - 原本的实现会内存爆炸（需要500GB），需要特殊处理一下
    - 这一步耗时约20min
- edit_weight很快
- generate，约需要2hr
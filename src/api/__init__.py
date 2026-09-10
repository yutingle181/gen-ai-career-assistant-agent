"""FastAPI 服务层：只做协议转换与持久化，不含业务规则。

注意：不要在包 __init__ 中反向 import main（会形成 import 循环，
导致 uvicorn 加载的 app 在路由注册前被创建、业务路由丢失）。
入口请用 `src.api.main:app`。
"""

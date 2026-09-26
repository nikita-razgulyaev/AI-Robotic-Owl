import asyncio
from modules.arbiter import arbiter, Priority

async def test():
    print(await arbiter.request(Priority.INDEPENDENT, 5, "test_idle"))
    print(await arbiter.request(Priority.INDEPENDENT, 5, "test_idle2"))
    print(arbiter.can_preempt(Priority.INDEPENDENT))
    print(arbiter.can_preempt(Priority.ENVIRONMENT))
    print(await arbiter.request(Priority.INDEPENDENT, 5, "should_fail"))

asyncio.run(test())
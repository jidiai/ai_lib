from registry import registry

@registry.registered(registry.STOPPER,"win_rate_stopper")
class WinRateStopper:
    def __init__(self,**kwargs):
        self.min_win_rate=kwargs["min_win_rate"]
        self.max_steps=kwargs["max_steps"]
    
    def stop(self, **kwargs):
        step=kwargs["step"]
        win_rate=kwargs["win_rate"]
        if step>=self.max_steps or win_rate>=self.min_win_rate:
            return True
        return False
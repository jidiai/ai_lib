'''
TODO(jh): Fault tolerance

Server: League Manager
    We use a linear task scheduler to delegate tasks to different workers.
    When a task is finished, the worker submmits results of the task and waits for a new task.
    TODO(jh): Maybe support task cancellation later.

Clients: League Worker
    main1
    main2
    exploiter
    league exploiter
    ...
    
Communications:
    Currently, we use TCP to communicate between server and clients.
    Redis maybe a good choice for out of ray communication, for its persistent storage(fault tolerence) and good scability.
'''

.header on

select distinct
trial.id as 'trial.id',
/*trial.name as 'trial.name',*/
trial.created as 'trial.created',
timer.id as 'timer.id',
timer.parent as 'timer.parent',
/*timer.name as 'timer.name',*/
timer.short_name as 'timer.short_name',
timer.timergroup as 'timer.timergroup',
metric.name as 'metric.name',
thread.node_rank as 'thread.node',
thread.thread_rank as 'thread.thread',
timer_value.value as 'timer_value.value'
from trial
left outer join timer on timer.trial = trial.id
left outer join metric on metric.trial = trial.id
left outer join thread on thread.trial = trial.id
left outer join timer_value on timer_value.timer = timer.id and timer_value.metric = metric.id and timer_value.thread = thread.id
where timer_value.value is not null
;
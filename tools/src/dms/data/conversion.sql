alter table metric add column trial int;
update metric set trial = (select xml_file.trial from xml_file where xml_file.metric = metric.id);
drop table xml_file;
alter table metric alter column trial set NOT NULL;

alter table application drop column experiment_table_name;
alter table experiment drop column trial_table_name;

drop view function_interval CASCADE;
drop view function_interval2 CASCADE;
drop view function_interval1 CASCADE;
drop view function_detail CASCADE;
drop view function_detail1 CASCADE;
drop view function_trial_experiment_view CASCADE;
drop view function_trial_view CASCADE;

drop INDEX interval_loc_f_m_n_c_t_index;
drop INDEX interval_mean_function_metric_index;
drop INDEX interval_total_function_metric_index;
drop INDEX interval_loc_function_metric_index;
drop INDEX function_trial_index;
drop INDEX trial_experiment_index;
drop INDEX experiment_application_index;

alter table function rename to interval_event;
alter table interval_location_profile rename column function to interval_event;
alter table interval_total_summary rename column function to interval_event;
alter table interval_mean_summary rename column function to interval_event;

alter table user_event rename to atomic_event;
alter table atomic_location_profile rename column user_event to atomic_event;

alter table experiment add FOREIGN KEY(application) REFERENCES application(id) ON DELETE NO ACTION ON UPDATE NO ACTION;
alter table trial add FOREIGN KEY(experiment) REFERENCES experiment(id) ON DELETE NO ACTION ON UPDATE NO ACTION;
alter table metric add FOREIGN KEY(trial) REFERENCES trial(id) ON DELETE NO ACTION ON UPDATE NO ACTION;
alter table interval_event add FOREIGN KEY(trial) REFERENCES trial(id) ON DELETE NO ACTION ON UPDATE NO ACTION;
alter table atomic_event add FOREIGN KEY(trial) REFERENCES trial(id) ON DELETE NO ACTION ON UPDATE NO ACTION;
alter table interval_location_profile add FOREIGN KEY(interval_event) REFERENCES interval_event(id) ON DELETE NO ACTION ON UPDATE NO ACTION;
alter table interval_total_summary add FOREIGN KEY(interval_event) REFERENCES interval_event(id) ON DELETE NO ACTION ON UPDATE NO ACTION;
alter table interval_mean_summary add FOREIGN KEY(interval_event) REFERENCES interval_event(id) ON DELETE NO ACTION ON UPDATE NO ACTION;
alter table atomic_location_profile add FOREIGN KEY(atomic_event) REFERENCES atomic_event(id) ON DELETE NO ACTION ON UPDATE NO ACTION;

CREATE INDEX interval_event_trial_index on interval_event (trial);
CREATE INDEX interval_loc_interval_event_metric_index on interval_location_profile (interval_event, metric);
CREATE INDEX interval_total_interval_event_metric_index on interval_total_summary (interval_event, metric);
CREATE INDEX interval_mean_interval_event_metric_index on interval_mean_summary (interval_event, metric);
CREATE INDEX interval_loc_f_m_n_c_t_index on interval_location_profile (interval_event, metric, node, context, thread);

vacuum verbose analyze;

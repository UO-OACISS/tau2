



CREATE TABLE application_temp (
    id                      SERIAL     NOT NULL        PRIMARY KEY,
    name                    TEXT,
    version		    TEXT,
    description             TEXT, 
    language                TEXT,
    paradigm                TEXT,
    usage_text              TEXT,
    execution_options       TEXT,
    userdata                TEXT
);

CREATE TABLE experiment_temp (
    id                      SERIAL      NOT NULL	PRIMARY KEY,
    application             INT         NOT NULL,
    name                    TEXT,
    system_name             TEXT,
    system_machine_type     TEXT,
    system_arch             TEXT,
    system_os               TEXT,
    system_memory_size      TEXT,
    system_processor_amt    TEXT,
    system_l1_cache_size    TEXT,
    system_l2_cache_size    TEXT,
    system_userdata         TEXT,
    compiler_cpp_name       TEXT,
    compiler_cpp_version    TEXT,
    compiler_cc_name        TEXT,
    compiler_cc_version     TEXT,
    compiler_java_dirpath   TEXT,
    compiler_java_version   TEXT,
    compiler_userdata       TEXT,
    configure_prefix        TEXT,
    configure_arch          TEXT,
    configure_cpp           TEXT,
    configure_cc            TEXT,
    configure_jdk           TEXT,
    configure_profile       TEXT,
    configure_userdata      TEXT,
    userdata                TEXT,
    FOREIGN KEY(application) REFERENCES application_temp(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE trial_temp (
    id                      SERIAL	NOT NULL	PRIMARY KEY,
    name                    TEXT,
    experiment              INT         NOT NULL,
    time                    TIMESTAMP   WITHOUT TIME ZONE,
    problem_definition      TEXT,
    node_count              INT,
    contexts_per_node       INT,
    threads_per_context     INT,
    userdata                TEXT,
    FOREIGN KEY(experiment) REFERENCES experiment_temp(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE metric_temp (
    id                      SERIAL      NOT NULL	PRIMARY KEY,
    name                    TEXT        NOT NULL,
    trial                   INT		NOT NULL,
    FOREIGN KEY(trial) REFERENCES trial_temp(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE interval_event_temp (
    id                      SERIAL      NOT NULL	PRIMARY KEY,
    trial                   INT         NOT NULL,
    name                    TEXT        NOT NULL,
    group_name              TEXT,
    source_file		    TEXT,
    line_number		    INT,
    line_number_end	    INT,
    FOREIGN KEY(trial) REFERENCES trial_temp(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE atomic_event_temp (
    id                      SERIAL      NOT NULL	PRIMARY KEY,
    trial                   INT         NOT NULL,
    name                    TEXT        NOT NULL,
    group_name              TEXT,
    source_file		    TEXT,
    line_number		    INT,
    FOREIGN KEY(trial) REFERENCES trial_temp(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE interval_location_profile_temp (
    interval_event          INT         NOT NULL,
    node                    INT         NOT NULL,             
    context                 INT         NOT NULL,
    thread                  INT         NOT NULL,
    metric                  INT		NOT NULL,
    inclusive_percentage    DOUBLE PRECISION,
    inclusive               DOUBLE PRECISION,
    exclusive_percentage    DOUBLE PRECISION,
    exclusive               DOUBLE PRECISION,
    call                    DOUBLE PRECISION,
    subroutines             DOUBLE PRECISION,
    inclusive_per_call      DOUBLE PRECISION,
    sum_exclusive_squared   DOUBLE PRECISION,
    FOREIGN KEY(interval_event) REFERENCES interval_event_temp(id) ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY(metric) REFERENCES metric_temp(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE atomic_location_profile_temp (
    atomic_event            INT         NOT NULL,
    node                    INT         NOT NULL,             
    context                 INT         NOT NULL,
    thread                  INT         NOT NULL,
    sample_count            INT,         
    maximum_value           DOUBLE PRECISION,
    minimum_value           DOUBLE PRECISION,
    mean_value              DOUBLE PRECISION,
    standard_deviation	    DOUBLE PRECISION,
    FOREIGN KEY(atomic_event) REFERENCES atomic_event_temp(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE interval_total_summary_temp (
    interval_event          INT         NOT NULL,
    metric                  INT		NOT NULL,
    inclusive_percentage    DOUBLE PRECISION,
    inclusive               DOUBLE PRECISION,
    exclusive_percentage    DOUBLE PRECISION,
    exclusive               DOUBLE PRECISION,
    call                    DOUBLE PRECISION,
    subroutines             DOUBLE PRECISION,
    inclusive_per_call      DOUBLE PRECISION,
    sum_exclusive_squared   DOUBLE PRECISION,
    FOREIGN KEY(interval_event) REFERENCES interval_event(id) ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY(metric) REFERENCES metric(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE interval_mean_summary_temp (
    interval_event          INT         NOT NULL,
    metric                  INT		NOT NULL,
    inclusive_percentage    DOUBLE PRECISION,
    inclusive               DOUBLE PRECISION,
    exclusive_percentage    DOUBLE PRECISION,
    exclusive               DOUBLE PRECISION,
    call                    DOUBLE PRECISION,
    subroutines             DOUBLE PRECISION,
    inclusive_per_call      DOUBLE PRECISION,
    sum_exclusive_squared   DOUBLE PRECISION,
	FOREIGN KEY(interval_event) REFERENCES interval_event_temp(id) ON DELETE NO ACTION ON UPDATE NO ACTION,
	FOREIGN KEY(metric) REFERENCES metric_temp(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);




insert into application_temp (select id, name, version, description, language, paradigm, usage_text, execution_options, userdata from application);

insert into experiment_temp (select id,
    application          ,
    name                    ,
    system_name             ,
    system_machine_type     ,
    system_arch             ,
    system_os               ,
    system_memory_size      ,
    system_processor_amt    ,
    system_l1_cache_size    ,
    system_l2_cache_size    ,
    system_userdata         ,
    compiler_cpp_name       ,
    compiler_cpp_version    ,
    compiler_cc_name        ,
    compiler_cc_version     ,
    compiler_java_dirpath   ,
    compiler_java_version   ,
    compiler_userdata       ,
    configure_prefix        ,
    configure_arch          ,
    configure_cpp           ,
    configure_cc            ,
    configure_jdk           ,
    configure_profile       ,
    configure_userdata      ,
    userdata      
from experiment);


insert into trial_temp (select id, name, experiment, time, problem_definition, node_count, contexts_per_node, threads_per_context, userdata from trial);

insert into metric_temp (select id, name, trial from metric);

insert into interval_event_temp (select id, trial, name, group_name from interval_event);

insert into atomic_event_temp (select id, trial, name, group_name from atomic_event);
insert into interval_location_profile_temp (select interval_event, node, context, thread, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call from interval_location_profile);

insert into atomic_location_profile_temp (select atomic_event, node, context, thread, sample_count, maximum_value, minimum_value, mean_value, standard_deviation from atomic_location_profile);

insert into interval_total_summary_temp (select interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call from interval_total_summary);

insert into interval_mean_summary_temp (select interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call from interval_mean_summary);




select setval('application_temp_id_seq', (nextval('application_id_seq') - 1));
select setval('experiment_temp_id_seq', (nextval('experiment_id_seq') - 1));
select setval('trial_temp_id_seq', (nextval('trial_id_seq') - 1));
select setval('metric_temp_id_seq', (nextval('metric_id_seq') - 1));
select setval('interval_event_temp_id_seq', (nextval('interval_event_id_seq') - 1));
select setval('atomic_event_temp_id_seq', (nextval('atomic_event_id_seq') - 1));



drop INDEX experiment_application_index;
drop INDEX trial_experiment_index;
drop INDEX interval_event_trial_index;
drop INDEX interval_loc_interval_event_metric_index;
drop INDEX interval_total_interval_event_metric_index;
drop INDEX interval_mean_interval_event_metric_index;
drop INDEX interval_loc_f_m_n_c_t_index;

drop table interval_mean_summary CASCADE;
drop table interval_total_summary CASCADE;
drop table interval_location_profile CASCADE;
drop table interval_event CASCADE;
drop table atomic_location_profile CASCADE;
drop table atomic_event CASCADE;
drop table metric CASCADE;
drop table trial CASCADE;
drop table experiment CASCADE;
drop table application CASCADE;



CREATE TABLE version (
    version		    TEXT
);

INSERT INTO version values('2.13.7');



CREATE TABLE application (
    id                      SERIAL	NOT NULL	PRIMARY KEY,
    name                    TEXT,
    version		    TEXT,
    description             TEXT, 
    language                TEXT,
    paradigm                TEXT,
    usage_text              TEXT,
    execution_options       TEXT,
    userdata                TEXT
);

CREATE TABLE experiment (
    id                      SERIAL      NOT NULL	PRIMARY KEY,
    application             INT         NOT NULL,
    name                    TEXT,
    system_name             TEXT,
    system_machine_type     TEXT,
    system_arch             TEXT,
    system_os               TEXT,
    system_memory_size      TEXT,
    system_processor_amt    TEXT,
    system_l1_cache_size    TEXT,
    system_l2_cache_size    TEXT,
    system_userdata         TEXT,
    compiler_cpp_name       TEXT,
    compiler_cpp_version    TEXT,
    compiler_cc_name        TEXT,
    compiler_cc_version     TEXT,
    compiler_java_dirpath   TEXT,
    compiler_java_version   TEXT,
    compiler_userdata       TEXT,
    configure_prefix        TEXT,
    configure_arch          TEXT,
    configure_cpp           TEXT,
    configure_cc            TEXT,
    configure_jdk           TEXT,
    configure_profile       TEXT,
    configure_userdata      TEXT,
    userdata                TEXT,
    FOREIGN KEY(application) REFERENCES application(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE trial (
    id                      SERIAL	NOT NULL	PRIMARY KEY,
    name                    TEXT,
    experiment              INT         NOT NULL,
    time                    TIMESTAMP   WITHOUT TIME ZONE,
    problem_definition      TEXT,
    node_count              INT,
    contexts_per_node       INT,
    threads_per_context     INT,
    userdata                TEXT,
    FOREIGN KEY(experiment) REFERENCES experiment(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE metric (
    id                      SERIAL      NOT NULL	PRIMARY KEY,
    name                    TEXT        NOT NULL,
    trial                   INT		NOT NULL,
    FOREIGN KEY(trial) REFERENCES trial(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE interval_event (
    id                      SERIAL      NOT NULL	PRIMARY KEY,
    trial                   INT         NOT NULL,
    name                    TEXT        NOT NULL,
    group_name              TEXT,
    source_file		    TEXT,
    line_number		    INT,
    line_number_end	    INT,
    FOREIGN KEY(trial) REFERENCES trial(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE atomic_event (
    id                      SERIAL      NOT NULL	PRIMARY KEY,
    trial                   INT         NOT NULL,
    name                    TEXT        NOT NULL,
    group_name              TEXT,
    source_file		    TEXT,
    line_number		    INT,
    FOREIGN KEY(trial) REFERENCES trial(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE interval_location_profile (
    interval_event          INT         NOT NULL,
    node                    INT         NOT NULL,             
    context                 INT         NOT NULL,
    thread                  INT         NOT NULL,
    metric                  INT		NOT NULL,
    inclusive_percentage    DOUBLE PRECISION,
    inclusive               DOUBLE PRECISION,
    exclusive_percentage    DOUBLE PRECISION,
    exclusive               DOUBLE PRECISION,
    call                    DOUBLE PRECISION,
    subroutines             DOUBLE PRECISION,
    inclusive_per_call      DOUBLE PRECISION,
    sum_exclusive_squared   DOUBLE PRECISION,
    FOREIGN KEY(interval_event) REFERENCES interval_event(id) ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY(metric) REFERENCES metric(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE atomic_location_profile (
    atomic_event            INT         NOT NULL,
    node                    INT         NOT NULL,             
    context                 INT         NOT NULL,
    thread                  INT         NOT NULL,
    sample_count            INT,         
    maximum_value           DOUBLE PRECISION,
    minimum_value           DOUBLE PRECISION,
    mean_value              DOUBLE PRECISION,
    standard_deviation	    DOUBLE PRECISION,
    FOREIGN KEY(atomic_event) REFERENCES atomic_event(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE interval_total_summary (
    interval_event          INT         NOT NULL,
    metric                  INT		NOT NULL,
    inclusive_percentage    DOUBLE PRECISION,
    inclusive               DOUBLE PRECISION,
    exclusive_percentage    DOUBLE PRECISION,
    exclusive               DOUBLE PRECISION,
    call                    DOUBLE PRECISION,
    subroutines             DOUBLE PRECISION,
    inclusive_per_call      DOUBLE PRECISION,
    sum_exclusive_squared   DOUBLE PRECISION,
    FOREIGN KEY(interval_event) REFERENCES interval_event(id) ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY(metric) REFERENCES metric(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE interval_mean_summary (
    interval_event          INT         NOT NULL,
    metric                  INT		NOT NULL,
    inclusive_percentage    DOUBLE PRECISION,
    inclusive               DOUBLE PRECISION,
    exclusive_percentage    DOUBLE PRECISION,
    exclusive               DOUBLE PRECISION,
    call                    DOUBLE PRECISION,
    subroutines             DOUBLE PRECISION,
    inclusive_per_call      DOUBLE PRECISION,
    sum_exclusive_squared   DOUBLE PRECISION,
	FOREIGN KEY(interval_event) REFERENCES interval_event(id) ON DELETE NO ACTION ON UPDATE NO ACTION,
	FOREIGN KEY(metric) REFERENCES metric(id) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE INDEX experiment_application_index on experiment (application);
CREATE INDEX trial_experiment_index on trial (experiment);
CREATE INDEX interval_event_trial_index on interval_event (trial);
CREATE INDEX interval_loc_interval_event_metric_index on interval_location_profile (interval_event, metric);
CREATE INDEX interval_total_interval_event_metric_index on interval_total_summary (interval_event, metric);
CREATE INDEX interval_mean_interval_event_metric_index on interval_mean_summary (interval_event, metric);
CREATE INDEX interval_loc_f_m_n_c_t_index on interval_location_profile (interval_event, metric, node, context, thread);




insert into application (select id, name, version, description, language, paradigm, usage_text, execution_options, userdata from application_temp);


insert into experiment (select id,
    application          ,
    name                    ,
    system_name             ,
    system_machine_type     ,
    system_arch             ,
    system_os               ,
    system_memory_size      ,
    system_processor_amt    ,
    system_l1_cache_size    ,
    system_l2_cache_size    ,
    system_userdata         ,
    compiler_cpp_name       ,
    compiler_cpp_version    ,
    compiler_cc_name        ,
    compiler_cc_version     ,
    compiler_java_dirpath   ,
    compiler_java_version   ,
    compiler_userdata       ,
    configure_prefix        ,
    configure_arch          ,
    configure_cpp           ,
    configure_cc            ,
    configure_jdk           ,
    configure_profile       ,
    configure_userdata      ,
    userdata      
from experiment_temp);



insert into trial (select id, name, experiment, time, problem_definition, node_count, contexts_per_node, threads_per_context, userdata from trial_temp);


insert into metric (select id, name, trial from metric_temp);


insert into interval_event (select id, trial, name, group_name from interval_event_temp);

insert into atomic_event (select id, trial, name, group_name from atomic_event_temp);
insert into interval_location_profile (select interval_event, node, context, thread, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call from interval_location_profile_temp);

insert into atomic_location_profile (select atomic_event, node, context, thread, sample_count, maximum_value, minimum_value, mean_value, standard_deviation from atomic_location_profile_temp);

insert into interval_total_summary (select interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call, sum_exclusive_squared from interval_total_summary_temp);

insert into interval_mean_summary (select interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call, sum_exclusive_squared from interval_mean_summary_temp);



select setval('application_id_seq', (nextval('application_temp_id_seq')));
select setval('experiment_id_seq', (nextval('experiment_temp_id_seq')));
select setval('trial_id_seq', (nextval('trial_temp_id_seq')));
select setval('metric_id_seq', (nextval('metric_temp_id_seq')));
select setval('interval_event_id_seq', (nextval('interval_event_temp_id_seq')));
select setval('atomic_event_id_seq', (nextval('atomic_event_temp_id_seq')));

drop table interval_mean_summary_temp CASCADE;
drop table interval_total_summary_temp CASCADE;
drop table interval_location_profile_temp CASCADE;
drop table interval_event_temp CASCADE;

drop table atomic_location_profile_temp CASCADE;
drop table atomic_event_temp CASCADE;


drop table metric_temp CASCADE;
drop table trial_temp CASCADE;
drop table experiment_temp CASCADE;
drop table application_temp CASCADE;

vacuum analyze;

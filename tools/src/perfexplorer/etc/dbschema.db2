create table @DATABASE_PREFIX@analysis_settings (
	id					INT NOT NULL PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
	application			INTEGER			NOT NULL,
	experiment			INTEGER			NOT NULL,
	trial				INTEGER			NOT NULL,
	metric				INTEGER			NULL,
	method				VARCHAR(255)	NOT NULL,
	dimension_reduction	VARCHAR(255)	NOT NULL,
	normalization		VARCHAR(255)	NOT NULL,
	FOREIGN KEY (application) REFERENCES @DATABASE_PREFIX@application(id) 
		ON DELETE CASCADE,
	FOREIGN KEY (experiment) REFERENCES @DATABASE_PREFIX@experiment(id) 
		ON DELETE CASCADE,
	FOREIGN KEY (trial) REFERENCES @DATABASE_PREFIX@trial(id) 
		ON DELETE CASCADE,
	FOREIGN KEY (metric) REFERENCES @DATABASE_PREFIX@metric(id) 
		ON DELETE CASCADE
);

create table @DATABASE_PREFIX@analysis_result (
	id					INT NOT NULL PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
	analysis_settings	INTEGER			NOT NULL,
	description			VARCHAR(255)	NOT NULL,
	thumbnail_size		INTEGER			NULL,
	image_size			INTEGER			NULL,
	thumbnail			BYTEA			NULL,
	image				BYTEA			NULL,
	result_type			INTEGER			NOT NULL
);

create table @DATABASE_PREFIX@trial_view (
	id					INT NOT NULL PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
	parent				INTEGER			NULL,
	name				VARCHAR(4000)	NOT NULL,
	table_name			VARCHAR(4000)	NOT NULL,
	column_name			VARCHAR(4000)	NOT NULL,
	operator			VARCHAR(4000)	NOT NULL,
	value				VARCHAR(4000)	NOT NULL,
	FOREIGN KEY (parent) REFERENCES @DATABASE_PREFIX@trial_view(id) 
		ON DELETE CASCADE
);

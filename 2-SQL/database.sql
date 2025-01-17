---CREATE TABLE---
CREATE TABLE ACCOUNT(
	user_id SERIAL PRIMARY KEY,
	username VARCHAR(50) UNIQUE NOT NULL,
	password VARCHAR(50) NOT NULL,
	email VARCHAR(250) UNIQUE NOT NULL,
	created_on TIMESTAMP NOT NULL,
	last_login TIMESTAMP
)
------
CREATE TABLE job(
	job_id SERIAL PRIMARY KEY,
	job_name VARCHAR(200) UNIQUE NOT NULL
)
----
--CREATE TABLE account_job(
	user_id INTEGER REFERENCES account(user_id)
)
----
CREATE TABLE account_job1(
	user_id INTEGER REFERENCES account(user_id),
	job_id INTEGER REFERENCES job(job_id),
	hire_date TIMESTAMP
)
------
SELECT * FROM account
----INSERT VALUES QUERY---
INSERT INTO account(username,password,email,created_on)
VALUES
('Jose','password','jos@mail.com',CURRENT_TIMESTAMP)
----
INSERT INTO job(job_name)
VALUES('Astronaut')
----
SELECT * FROM job
----
INSERT INTO job(job_name)
VALUES('President')
-----
SELECT * FROM job
----
INSERT INTO account_job1(user_id,job_id,hire_date)
VALUES(1,1,CURRENT_TIMESTAMP)
----
SELECT * FROM account_job1
---ERROR---
INSERT INTO account_job1(user_id,job_id,hire_date)
VALUES(10,10,CURRENT_TIMESTAMP)
----
INSERT INTO account(username,password,email,created_on)
VALUES
('Ram','root','ram1@sanjivani.org.in',CURRENT_TIMESTAMP)
----
SELECT * FROM account
---
INSERT INTO job(job_name)
VALUES('Data Scientist')
SELECT * FROM job
----
---UPDATE---
SELECT * FROM account
---
UPDATE account
SET last_login=CURRENT_TIMESTAMP
WHERE last_login IS NULL;
---
UPDATE account
SET last_login=created_on
----
SELECT * FROM account_job1
--
UPDATE account_job1
SET hire_date=account.created_on
FROM account
WHERE account_job1.user_id=account.user_id
----
---10-07-2024---
---DELETE CLAUSE----
SELECT * FROM job
--
INSERT INTO job(job_name)
VALUES('cowboy')
SELECT * FROM job
--
DELETE FROM job
WHERE job_name='cowboy'
RETURNING job_id,job_name
SELECT * FROM job
--
---ALTER CLAUSE---
CREATE TABLE information(
	info_id SERIAL PRIMARY KEY,
	title VARCHAR(500) NOT NULL,
	person VARCHAR(50) NOT NULL UNIQUE
)
--
SELECT * FROM information
--
ALTER TABLE information
RENAME TO new_info
--
SELECT * FROM information ##ERROR
--
SELECT * FROM new_info
--
ALTER TABLE new_info
RENAME COLUMN person TO people
--
SELECT * FROM new_info
--##error
INSERT INTO new_info(title) 
VALUES('some new title')
--
ALTER TABLE new_info
ALTER COLUMN people DROP NOT NULL
--
SELECT * FROM new_info
----DROP CLAUSE--
SELECT * FROM new_info
--
ALTER TABLE new_info
DROP COLUMN people
SELECT * FROM new_info
---
ALTER TABLE new_info
DROP COLUMN people
SELECT * FROM new_info
--
ALTER TABLE new_info
DROP COLUMN IF EXISTS people
---CHECK CONSTRAINT---
CREATE TABLE employees(
	emp_id SERIAL PRIMARY KEY,
	first_name VARCHAR(50) NOT NULL,
	last_name VARCHAR(50) NOT NULL,
	birthdate DATE CHECK (birthdate > '1900-01-01'),
	hire_date DATE CHECK (hire_date > birthdate),
	salary INTEGER CHECK (salary > 0)
)
--
INSERT INTO employees(
	first_name,
	last_name,
	birthdate,
	hire_date,
	salary
)
VALUES
('Jose',
 'Portilla',
 '1990-11-03',
 '2010-01-01',
 100
)
--
INSERT INTO employees(
	first_name,
	last_name,
	birthdate,
	hire_date,
	salary
)
VALUES
('Samny',
 'Smith',
 '1990-11-03',
 '2010-01-01',
 100
)
---

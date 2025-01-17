---12-07-2024
SHOW TIMEZONE
--
CREATE TABLE timezone(
	ts TIMESTAMP WITHOUT TIME ZONE,
	tz TIMESTAMP WITH TIME ZONE
)
--
SELECT * FROM timezone
--
INSERT INTO TIMEZONE VALUES(
	TIMESTAMP WITHOUT TIME ZONE '2023-03-07 10:50',
	TIMESTAMP WITH TIME ZONE '2023-03-07 10:50'
)
SELECT * FROM timezone
--CURRENT DATE 
SELECT now()::date;
SELECT CURRENT_DATE
--
SELECT TO_CHAR(CURRENT_DATE,'dd/mm/yy')
--
SELECT TO_CHAR(CURRENT_DATE,'DDD')
--
SELECT TO_CHAR(CURRENT_DATE,'ww')
--CALCULATE AGE
SELECT AGE(date'2004-07-10') 
--
SELECT AGE(date'')
--DISPLAY ONLY DAY
SELECT EXTRACT (DAY FROM date'1992/11/13')AS DAY
	--13
--DISPLAY ONLY MONTH
SELECT EXTRACT (MONTH FROM date'1992/11/13')AS MONTH
	--11
--DISPLAY ONLY YEAR
SELECT EXTRACT (YEAR FROM date'1992/11/13')AS YEAR
	--1992
--timestamp with timezone
SELECT DATE_TRUNC('year',date'1992/11/13')
---get me all the employees above 60,use the appropriate date functions.
SELECT AGE(birth_date),*FROM employees
WHERE(
	EXTRACT(YEAR FROM AGE(birth_date))
)>60;
---
SELECT count(birth_date) FROM employees
WHERE birth_date<now()-interval '61 years'
---
SELECT count(emp_no) FROM employees
WHERE EXTRACT(MONTH FROM hire_date)=2;
--
SELECT count(emp_no) FROM employees
WHERE EXTRACT(MONTH FROM birth_date)=11;
--
SELECT MAX(AGE(birth_date)) FROM employees;
--
SELECT MAX(salary) FROM salaries
  --158220
--
SELECT * FROM salaries
--
SELECT *,
   MAX(salary) 
   FROM salaries
	--ERROR
---
SELECT *,
   MAX(salary) OVER()
   FROM salaries
--TAKE 100 ROWS ONLY
SELECT *,
   MAX(salary) OVER()
   FROM salaries
LIMIT 100
--display row where salary is <70000
SELECT *,
   MAX(salary) OVER()
   FROM salaries
WHERE salary<70000
--
SELECT *,
   AVG(salary) OVER()
   FROM salaries
--
SELECT *,
 d.dept_name,
 AVG(salary) OVER()
FROM salaries
JOIN dept_emp AS de USING (emp_no)
JOIN departments AS d USING (dept_no)
--
SELECT *,
 d.dept_name,
 AVG(salary) OVER(
	PARTITION BY d.dept_name  ---avg salary dept vise
	)
FROM salaries
JOIN dept_emp AS de USING (emp_no)
JOIN departments AS d USING (dept_no)
---
SELECT *,
 AVG(salary) OVER(
	PARTITION BY d.dept_name  
	)
FROM salaries
JOIN dept_emp AS de USING (emp_no)
JOIN departments AS d USING (dept_no)
--







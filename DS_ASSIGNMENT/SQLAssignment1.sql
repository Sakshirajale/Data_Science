--Assignment-1:
create table company(
	log_id SERIAL PRIMARY KEY,
	machine_id int,
	production_date date,
	production_shift varchar(50),
	products_produced int,
	defects int,
	runtime int
);

insert into company
(
	machine_id,
	production_date,
	production_shift,
	products_produced,
	defects,
	runtime 
)
values(
1,'2024-06-01', 'Morning', 500, 5, 8.0)

insert into company
(
	machine_id,
	production_date,
	production_shift,
	products_produced,
	defects,
	runtime 
)
values(
1, '2024-06-01', 'Evening', 450, 3, 7.5)

insert into company
(
	machine_id,
	production_date,
	production_shift,
	products_produced,
	defects,
	runtime 
)
values(
2, '2024-06-01', 'Morning', 480, 2, 7.8)

insert into company
(
	machine_id,
	production_date,
	production_shift,
	products_produced,
	defects,
	runtime 
)
values(
2, '2024-06-01', 'Evening', 470, 4, 8.1)

insert into company
(
	machine_id,
	production_date,
	production_shift,
	products_produced,
	defects,
	runtime 
)
values(
3, '2024-06-01', 'Morning', 510, 6, 8.2)

insert into company
(
	machine_id,
	production_date,
	production_shift,
	products_produced,
	defects,
	runtime 
)
values(
3, '2024-06-01', 'Evening', 520, 1, 7.9)

insert into company
(
	machine_id,
	production_date,
	production_shift,
	products_produced,
	defects,
	runtime 
)
values(
1, '2024-06-02', 'Morning', 490, 3, 8.0)

insert into company
(
	machine_id,
	production_date,
	production_shift,
	products_produced,
	defects,
	runtime 
)
values(
1, '2024-06-02', 'Evening', 460, 2, 7.6)
	
insert into company
(
	machine_id,
	production_date,
	production_shift,
	products_produced,
	defects,
	runtime 
)
values(
2, '2024-06-02', 'Morning', 475, 1, 7.9)
	
insert into company
(
	machine_id,
	production_date,
	production_shift,
	products_produced,
	defects,
	runtime 
)
values(
2, '2024-06-02', 'Evening', 465, 5, 8.3)
	
insert into company
(
	machine_id,
	production_date,
	production_shift,
	products_produced,
	defects,
	runtime 
)
values(
3, '2024-06-02', 'Morning', 505, 4, 8.0)
	
insert into company
(
	machine_id,
	production_date,
	production_shift,
	products_produced,
	defects, 
	runtime 
)
values(
3, '2024-06-02', 'Evening', 515, 3, 8.2)

SELECT * FROM company;

SELECT MAX(defects) FROM company;

SELECT AVG(runtime) FROM company;

--Assignment-2:
create table student(
	grade_id SERIAL PRIMARY KEY,
	student_id int,
	course_id int,
	grade varchar(10),
	grade_date date
);


insert into student
(
	student_id,
	course_id,
	grade,
	grade_date
)
values(
1, 101, 85.5, '2024-01-15')

	
insert into student
(
	student_id,
	course_id,
	grade,
	grade_date
)
values(
1, 102, 78.0, '2024-02-20')

insert into student
(
	student_id,
	course_id,
	grade,
	grade_date
)
values(
2, 101, 92.0, '2024-01-15')

	
insert into student
(
	student_id,
	course_id,
	grade,
	grade_date
)
values(
2, 103, 88.5, '2024-03-10')

	
insert into student
(
	student_id,
	course_id,
	grade,
	grade_date
)
values(
3, 102, 74.0, '2024-02-20')

	
insert into student
(
	student_id,
	course_id,
	grade,
	grade_date
)
values(
3, 103, 81.5, '2024-03-10')

	
insert into student
(
	student_id,
	course_id,
	grade,
	grade_date
)
values(
4, 101, 67.0, '2024-01-15')

	
insert into student
(
	student_id,
	course_id,
	grade,
	grade_date
)
values(
4, 104, 90.0, '2024-04-05')

	
insert into student
(
	student_id,
	course_id,
	grade,
	grade_date
)
values(
5, 102, 85.0, '2024-02-20')

	
insert into student
(
	student_id,
	course_id,
	grade,
	grade_date
)
values(
5, 104, 87.0, '2024-04-05')

SELECT * FROM student;

--SELECT AVG(grade) FROM student;
--Assignment-3
create table customer(
	customer_id SERIAL PRIMARY KEY,
	first_name varchar(20),
	last_name varchar(20),
	email varchar(50),
	phone_number varchar(20),
	address varchar(100),
	city varchar(50),
	state varchar(50),
	zip_code int,
	plan_id int,
	last_call_date date
);


insert into customer(
	first_name,
	last_name,
	email,
	phone_number,
	address,
	city,
	state,
	zip_code,
	plan_id,
	last_call_date
)
values
('John', 'Doe', 'john.doe@example.com', '123-456-7890', '123 Elm St', 'Springfield', 'IL', '62701', 1, '2024-06-01')


insert into customer(
	first_name,
	last_name,
	email,
	phone_number,
	address,
	city,
	state,
	zip_code,
	plan_id,
	last_call_date
)
values
('Jane', 'Smith', 'jane.smith@example.com', '987-654-3210', '456 Oak St', 'Springfield', 'IL', '62701', 2, '2024-05-15')

insert into customer(
	first_name,
	last_name,
	email,
	phone_number,
	address,
	city,
	state,
	zip_code,
	plan_id,
	last_call_date
)
values
('Alice', 'Johnson', 'alice.johnson@example.com', '555-123-4567', '789 Pine St', 'Shelbyville', 'IL', '62565', 3, '2024-04-20')

insert into customer(
	first_name,
	last_name,
	email,
	phone_number,
	address,
	city,
	state,
	zip_code,
	plan_id,
	last_call_date
)
values
('Bob', 'Brown', 'bob.brown@example.com', '444-555-6666', '101 Maple St', 'Capital City', 'IL', '62702', 1, '2024-06-10')

insert into customer(
	first_name,
	last_name,
	email,
	phone_number,
	address,
	city,
	state,
	zip_code,
	plan_id,
	last_call_date
)
values
('Charlie', 'Davis', 'charlie.davis@example.com', '333-444-5555', '202 Cedar St', 'Springfield', 'IL', '62701', 2, '2024-03-30')

insert into customer(
	first_name,
	last_name,
	email,
	phone_number,
	address,
	city,
	state,
	zip_code,
	plan_id,
	last_call_date
)
values
('Diana', 'Evans', 'diana.evans@example.com', '222-333-4444', '303 Birch St', 'Shelbyville', 'IL', '62565', 3, '2024-06-20')

insert into customer(
	first_name,
	last_name,
	email,
	phone_number,
	address,
	city,
	state,
	zip_code,
	plan_id,
	last_call_date
)
values
('Ethan', 'Foster', 'ethan.foster@example.com', '111-222-3333', '404 Spruce St', 'Capital City', 'IL', '62702', 1, '2024-02-14')

insert into customer(
	first_name,
	last_name,
	email,
	phone_number,
	address,
	city,
	state,
	zip_code,
	plan_id,
	last_call_date
)
values
('Fiona', 'Garcia', 'fiona.garcia@example.com', '999-888-7777', '505 Ash St', 'Springfield', 'IL', '62701', 2, '2024-05-05')

insert into customer(
	first_name,
	last_name,
	email,
	phone_number,
	address,
	city,
	state,
	zip_code,
	plan_id,
	last_call_date
)
values
('George', 'Harris', 'george.harris@example.com', '888-777-6666', '606 Walnut St', 'Shelbyville', 'IL', '62565', 3, '2024-01-25')


insert into customer(
	first_name,
	last_name,
	email,
	phone_number,
	address,
	city,
	state,
	zip_code,
	plan_id,
	last_call_date
)
values
('Hannah', 'Irvine', 'hannah.irvine@example.com', '777-666-5555', '707 Hickory St', 'Capital City', 'IL', '62702', 1, '2024-06-22');

SELECT * FROM customer;

--Assignment-3 part(2)
create table product
(
	product_id SERIAL PRIMARY KEY,
	product_name varchar(20),
	category varchar(20),
	quantity int,
	price float,
	supplier varchar(30),
	last_restock_date date
);

insert into product
(
	product_name,
	category,
	quantity,
	price,
	supplier,
	last_restock_date
)
values
('Laptop', 'Electronics', 50, 799.99, 'TechSupplier Inc.', '2024-06-01')

insert into product
(
	product_name,
	category,
	quantity,
	price,
	supplier,
	last_restock_date
)
values
('Smartphone', 'Electronics', 150, 499.99, 'MobileDistributors Ltd.', '2024-05-25')

insert into product
(
	product_name,
	category,
	quantity,
	price,
	supplier,
	last_restock_date
)
values
('Desk Chair', 'Furniture', 80, 89.99, 'OfficeSupplies Co.', '2024-05-15')

insert into product
(
	product_name,
	category,
	quantity,
	price,
	supplier,
	last_restock_date
)
values
('Notebook', 'Stationery', 200, 2.99, 'PaperGoods Inc.', '2024-06-10')

insert into product
(
	product_name,
	category,
	quantity,
	price,
	supplier,
	last_restock_date
)
values
('Water Bottle', 'Accessories', 120, 9.99, 'Lifestyle Products', '2024-06-05')

insert into product
(
	product_name,
	category,
	quantity,
	price,
	supplier,
	last_restock_date
)
values
('Headphones', 'Electronics', 70, 149.99, 'TechSupplier Inc.', '2024-06-08')

insert into product
(
	product_name,
	category,
	quantity,
	price,
	supplier,
	last_restock_date
)
values
('Desk Lamp', 'Furniture', 60, 29.99, 'OfficeSupplies Co.', '2024-05-20')

insert into product
(
	product_name,
	category,
	quantity,
	price,
	supplier,
	last_restock_date
)
values
('Backpack', 'Accessories', 90, 49.99, 'TravelGear Ltd.', '2024-06-12')


insert into product
(
	product_name,
	category,
	quantity,
	price,
	supplier,
	last_restock_date
)
values
('Pen', 'Stationery', 300, 1.49, 'PaperGoods Inc.', '2024-06-03')

insert into product
(
	product_name,
	category,
	quantity,
	price,
	supplier,
	last_restock_date
)
values
('Monitor', 'Electronics', 40, 199.99, 'TechSupplier Inc.', '2024-06-15')

SELECT * FROM product;

SELECT * FROM products
WHERE quantity < 50;

--Assignment-4
create table patient
(
	patient_id SERIAL PRIMARY KEY,
	first_name varchar(20),
	last_name varchar(20),
	date_of_birth date,
	gender char(20),
	phone_number varchar(15),
	email varchar(40),
	address varchar(30),
	city char(20),
	state char(20),
	zip_code int,
	medical_history varchar(30),
	last_visit_date date	
);


insert into patient
(
	first_name,
	last_name,
	date_of_birth,
	gender,
	phone_number,
	email,
	address,
	city,
	state,
	zip_code,
	medical_history,
	last_visit_date
)
values
('John', 'Doe', '1980-01-15', 'Male','123-456-7890', 'john.doe@example.com', '123 Elm St','Springfield', 'IL', '62701', 'Hypertension', '2024-06-01')


insert into patient
(
	first_name,
	last_name,
	date_of_birth,
	gender,
	phone_number,
	email,
	address,
	city,
	state,
	zip_code,
	medical_history,
	last_visit_date
)
values
('Jane', 'Smith', '1990-02-20', 'Female', '987-654-3210', 'jane.smith@example.com', '456 Oak St', 'Springfield', 'IL', '62701', 'Diabetes', '2024-05-25')


insert into patient
(
	first_name,
	last_name,
	date_of_birth,
	gender,
	phone_number,
	email,
	address,
	city,
	state,
	zip_code,
	medical_history,
	last_visit_date
)
values
('Alice', 'Johnson', '1975-03-30', 'Female', '555-123-4567', 'alice.johnson@example.com', '789 Pine St', 'Shelbyville', 'IL', '62565', 'Asthma', '2024-06-10')

insert into patient
(
	first_name,
	last_name,
	date_of_birth,
	gender,
	phone_number,
	email,
	address,
	city,
	state,
	zip_code,
	medical_history,
	last_visit_date
)
values
('Bob', 'Brown', '1965-04-10', 'Male', '444-555-6666', 'bob.brown@example.com', '101 Maple St', 'Capital City', 'IL', '62702', 'High Cholesterol', '2024-05-15')


insert into patient
(
	first_name,
	last_name,
	date_of_birth,
	gender,
	phone_number,
	email,
	address,
	city,
	state,
	zip_code,
	medical_history,
	last_visit_date
)
values
('Charlie', 'Davis', '1985-05-20', 'Male', '333-444-5555', 'charlie.davis@example.com', '202 Cedar St', 'Springfield', 'IL', '62701', 'Allergies', '2024-06-05')

insert into patient
(
	first_name,
	last_name,
	date_of_birth,
	gender,
	phone_number,
	email,
	address,
	city,
	state,
	zip_code,
	medical_history,
	last_visit_date
)
values
('Diana', 'Evans', '2000-06-25', 'Female', '222-333-4444', 'diana.evans@example.com', '303 Birch St', 'Shelbyville', 'IL', '62565', 'Migraine', '2024-06-20')

insert into patient
(
	first_name,
	last_name,
	date_of_birth,
	gender,
	phone_number,
	email,
	address,
	city,
	state,
	zip_code,
	medical_history,
	last_visit_date
)
values
('Ethan', 'Foster', '1970-07-15', 'Male', '111-222-3333', 'ethan.foster@example.com', '404 Spruce St', 'Capital City', 'IL', '62702', 'Arthritis', '2024-06-12')


insert into patient
(
	first_name,
	last_name,
	date_of_birth,
	gender,
	phone_number,
	email,
	address,
	city,
	state,
	zip_code,
	medical_history,
	last_visit_date
)
values
('Fiona', 'Garcia', '1995-08-10', 'Female', '999-888-7777', 'fiona.garcia@example.com', '505 Ash St', 'Springfield', 'IL', '62701', 'Depression', '2024-05-30')


insert into patient
(
	first_name,
	last_name,
	date_of_birth,
	gender,
	phone_number,
	email,
	address,
	city,
	state,
	zip_code,
	medical_history,
	last_visit_date
)
values
('George', 'Harris', '1988-09-05', 'Male', '888-777-6666', 'george.harris@example.com', '606 Walnut St', 'Shelbyville', 'IL', '62565', 'Hypertension', '2024-04-25')


insert into patient
(
	first_name,
	last_name,
	date_of_birth,
	gender,
	phone_number,
	email,
	address,
	city,
	state,
	zip_code,
	medical_history,
	last_visit_date
)
values
('Hannah', 'Irvine', '1992-10-12', 'Female', '777-666-5555', 'hannah.irvine@example.com', '707 Hickory St', 'Capital City', 'IL', '62702', 'Diabetes', '2024-06-22')

SELECT * FROM patient;

SELECT * FROM patients
WHERE last_visit_date BETWEEN '2024-06-01' AND '2024-06-15';

--Assignment-5
create table transaction
(
	transaction_id SERIAL PRIMARY KEY,
	account_id int,
	transaction_date date,
	transaction_amount float,
	transaction_type char(20),
	merchant varchar(15),
	location varchar(30),
	status char(10)
);

insert into transaction
(
	account_id,
	transaction_date,
	transaction_amount,
	transaction_type,
	merchant,
	location,
	status
)
values
(1, '2024-06-01 10:00:00', 1000.00, 'Credit', 'Amazon', 'Online', 'Completed')

insert into transaction
(
	account_id,
	transaction_date,
	transaction_amount,
	transaction_type,
	merchant,
	location,
	status
)
values
(1, '2024-06-01 12:30:00', 500.00, 'Debit', 'Walmart', 'Springfield', 'Completed')


insert into transaction
(
	account_id,
	transaction_date,
	transaction_amount,
	transaction_type,
	merchant,
	location,
	status
)
values
(2, '2024-06-02 09:45:00', 15000.00, 'Credit', 'Apple Store', 'Chicago', 'Pending')


insert into transaction
(
	account_id,
	transaction_date,
	transaction_amount,
	transaction_type,
	merchant,
	location,
	status
)
values
(2, '2024-06-02 11:00:00', 200.00, 'Debit', 'Starbucks', 'Chicago', 'Completed')

insert into transaction
(
	account_id,
	transaction_date,
	transaction_amount,
	transaction_type,
	merchant,
	location,
	status
)
values
(3, '2024-06-03 14:15:00', 250.00, 'Debit', 'Target', 'Springfield', 'Completed')

insert into transaction
(
	account_id,
	transaction_date,
	transaction_amount,
	transaction_type,
	merchant,
	location,
	status
)
values
(3, '2024-06-03 16:20:00', 30000.00, 'Credit', 'Tesla', 'San Francisco', 'Pending')

insert into transaction
(
	account_id,
	transaction_date,
	transaction_amount,
	transaction_type,
	merchant,
	location,
	status
)
values
(4, '2024-06-04 08:30:00', 120.00, 'Debit', 'McDonalds', 'Springfield', 'Completed')

insert into transaction
(
	account_id,
	transaction_date,
	transaction_amount,
	transaction_type,
	merchant,
	location,
	status
)
values
(4, '2024-06-04 10:50:00', 6000.00, 'Credit', 'Best Buy', 'Chicago', 'Pending')
	


insert into transaction
(
	account_id,
	transaction_date,
	transaction_amount,
	transaction_type,
	merchant,
	location,
	status
)
values
(5, '2024-06-05 15:10:00', 70.00, 'Debit', 'CVS Pharmacy', 'Springfield', 'Completed')	
	

insert into transaction
(
	account_id,
	transaction_date,
	transaction_amount,
	transaction_type,
	merchant,
	location,
	status
)
values
(5, '2024-06-05 17:00:00', 22000.00, 'Credit', 'Louis Vuitton', 'New York', 'Pending');	

select * from transaction;

SELECT * FROM transactions
WHERE transaction_amount > 5000
AND transaction_date BETWEEN '2024-06-01 00:00:00' AND '2024-06-05 23:59:59';
--Assignment 2
CREATE TABLE grades (
    grade_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    course_id INT NOT NULL,
    grade VARCHAR(20) NOT NULL,
    grade_date DATE NOT NULL
);
--


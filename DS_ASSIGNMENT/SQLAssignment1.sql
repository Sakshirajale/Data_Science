--Assignment 2
CREATE TABLE grades (
    grade_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    course_id INT NOT NULL,
    grade VARCHAR(20) NOT NULL,
    grade_date DATE NOT NULL
);
--


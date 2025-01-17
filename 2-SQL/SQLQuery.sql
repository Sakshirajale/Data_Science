---03-07-2024
SELECT * FROM customer
WHERE first_name LIKE 'J%'
---
SELECT COUNT(*) FROM customer
WHERE first_name LIKE 'J%'
---
SELECT COUNT(*) FROM customer
WHERE first_name LIKE 'J%' AND last_name LIKE 'S%'
---
SELECT * FROM customer
WHERE first_name LIKE 'J%' AND last_name LIKE 'S%'
---
SELECT * FROM customer
WHERE first_name ILIKE 'j%' AND last_name ILIKE 'j%'
---
SELECT * FROM customer
WHERE first_name ILIKE 'J%' AND last_name ILIKE 'S%'
---
SELECT * FROM customer
WHERE first_name LIKE '%er%'
---
SELECT COUNT(*) FROM customer
WHERE first_name LIKE '%er%'
---
SELECT * FROM customer
WHERE first_name LIKE '%her%'
---
SELECT * FROM customer
WHERE first_name LIKE '_her%'
---
SELECT COUNT(*) FROM customer
WHERE first_name LIKE '_her%'
---04-07-2024
SELECT * FROM customer
WHERE first_name NOT LIKE '_her%'
---
SELECT COUNT(*) FROM customer
WHERE first_name NOT LIKE '_her%'
---
SELECT * FROM customer
WHERE first_name LIKE 'A%'
ORDER BY last_name
---
SELECT * FROM customer
WHERE first_name LIKE 'A%' AND last_name NOT LIKE 'B%'
ORDER BY last_name
---
SELECT COUNT(amount) FROM payment
WHERE amount>5;
---
SELECT * FROM actor
WHERE first_name LIKE 'P%'
---
SELECT COUNT(DISTINCT(district))
FROM address;
----
SELECT DISTINCT(district) FROM address;
---
SELECT COUNT(*) FROM film
WHERE rating='R'
AND replacement_cost BETWEEN 5 AND 15;
---
SELECT * FROM film
WHERE title LIKE '%Truman%';
---
SELECT COUNT(*) FROM film
WHERE title LIKE '%Truman%';
---
SELECT MIN(replacement_cost) FROM film;
---
SELECT MAX(replacement_cost) FROM film;
---
SELECT MAX(replacement_cost),MIN(replacement_cost) FROM film;

---05-07-2024
---Aggregate function---
SELECT * FROM payment
---
SELECT staff_id,customer_id,SUM(amount) FROM payment
GROUP BY staff_id,customer_id
---
SELECT staff_id,customer_id,SUM(amount) FROM payment
GROUP BY staff_id,customer_id
ORDER BY staff_id
---
SELECT staff_id,customer_id,SUM(amount) FROM payment
GROUP BY staff_id,customer_id
ORDER BY staff_id,customer_id
---
SELECT staff_id,customer_id,SUM(amount) FROM payment
GROUP BY staff_id,customer_id
ORDER BY SUM(amount)
---
SELECT * FROM payment 
---
SELECT DATE(payment_date) FROM payment
---amount spent datewise
SELECT DATE(payment_date),SUM(amount) FROM payment
GROUP BY DATE(payment_date)
---
SELECT DATE(payment_date),SUM(amount) FROM payment
GROUP BY DATE(payment_date)
ORDER BY SUM(amount)
---
SELECT DATE(payment_date),SUM(amount) FROM payment
GROUP BY DATE(payment_date)
ORDER BY SUM(amount) DESC
---
SELECT staff_id,COUNT(amount) FROM payment
GROUP BY staff_id
---
SELECT * FROM film
---
SELECT rating,AVG(replacement_cost) FROM film
GROUP BY rating
---
SELECT rating,ROUND(AVG(repacement_cost),)
---
SELECT customer_id,SUM(amount) FROM PAYMENT
GROUP BY customer_id
ORDER BY SUM(amount) DESC
LIMIT 5
---
SELECT customer_id FROM payment
GROUP BY customer_id

----HAVING clause---
--Having clause always use GROUP BY
SELECT customer_id,SUM(amount) FROM payment
WHERE customer_id NOT IN(184,87,477)
GROUP BY customer_id
---
SELECT customer_id,SUM(amount) FROM payment
GROUP BY customer_id
HAVING SUM(amount)>100
---08-07-2024----
SELECT customer_id, COUNT(*)FROM payment
GROUP BY customer_id
HAVING COUNT(*) >=40;
---
SELECT customer_id,SUM(amount) FROM payment
WHERE staff_id=2
GROUP BY customer_id
HAVING SUM(amount) >100
---#####----
SELECT amount AS rental_price FROM payment;
----
SELECT SUM(amount) AS net_revenue FROM payment;
----
SELECT COUNT(amount) AS num_transaction FROM payment;
----
SELECT COUNT(*) AS num_transaction FROM payment;
----
SELECT customer_id,SUM(amount)FROM payment
GROUP BY customer_id
----
SELECT customer_id,SUM(amount) AS total_spent FROM payment
GROUP BY customer_id
----
SELECT customer_id,SUM(amount)FROM payment
GROUP BY customer_id
HAVING SUM(amount) >100
----
SELECT customer_id,SUM(amount) AS tatal_spent FROM payment
GROUP BY customer_id
HAVING total_spent >100
---
SELECT customer_id,amount FROM payment
WHERE amount>2
---
SELECT customer_id,amount AS new_name FROM payment
WHERE new > 12
----
-----JOIN OPERATION----
SELECT * FROM payment
---
SELECT payment_id,payment.customer_id,first_name FROM payment
INNER JOIN customer
ON payment.customer_id=customer.customer_id
----
SELECT payment_id,payment.customer_id,first_name FROM customer
INNER JOIN payment
ON payment.customer_id=customer.customer_id
----FULL OUTER JOIN---
SELECT * FROM customer
FULL OUTER JOIN payment
ON customer.customer_id=payment.customer_id
-----
---LEFT OUTER JOIN---
SELECT * FROM INVENTORY
----
SELECT * FROM film 
----
SELECT film.film_id,title,inventory_id FROM film
LEFT JOIN inventory ON inventory.film_id = film.film_id
-----
SELECT film.film_id,title,inventory_id ,store_id FROM film
LEFT JOIN inventory ON inventory.film_id = film.film_id
---
SELECT film.film_id,title,inventory_id ,store_id FROM film
LEFT JOIN inventory ON inventory.film_id = film.film_id
WHERE inventory.film_id IS null
----
SELECT film.film_id,title,inventory_id ,store_id FROM inventory 
RIGHT JOIN film ON inventory.film_id = film.film_id
----
SELECT film.film_id,title,inventory_id ,store_id FROM inventory 
RIGHT JOIN film ON inventory.film_id = film.film_id
WHERE inventory.film_id IS null
---
----09-07-2024----
1)
SELECT * FROM customer
----
SELECT * FROM address
----
SELECT district,email FROM address
INNER JOIN customer ON 
address.address_id=customer.address_id
----
SELECT district,email FROM address
INNER JOIN customer ON 
address.address_id=customer.address_id
WHERE district= 'California'
----
2)
SELECT * FROM film
----
SELECT * FROM actor
----
SELECT * FROM film_actor
---
SELECT * FROM actor 
INNER JOIN film_actor
ON actor.actor_id=film_actor.actor_id
--------
SELECT * FROM actor 
INNER JOIN film_actor
ON actor.actor_id=film_actor.actor_id
INNER JOIN film
ON film_actor.film_id=film.film_id
-----
SELECT title,first_name,last_name FROM actor 
INNER JOIN film_actor
ON actor.actor_id=film_actor.actor_id
INNER JOIN film
ON film_actor.film_id=film.film_id
----
SELECT title,first_name,last_name FROM actor 
INNER JOIN film_actor
ON actor.actor_id=film_actor.actor_id
INNER JOIN film
ON film_actor.film_id=film.film_id
WHERE first_name='Nick' AND last_name='Wahlberg'
------``
----Primary Key And Foreign Key-----
SELECT * FROM payment



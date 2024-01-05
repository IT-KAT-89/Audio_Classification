create database soundDB;
use soundschema;
create table label
(
	lid int primary key,
    bezeichnung varchar(40)    
);

create table soundDatei
(
	sid int primary key AUTO_INCREMENT, 
    dateiPfad varchar(255),
    lid int,
    constraint fk_lbl foreign key (lid) references label (lid)
);

drop table soundDatei;

insert into label 
values(1, 'Bass_drum');
insert into label 
values(2, 'Snare_drum');
insert into label 
values(3, 'Hi-hat');

select * from label;
select * from soundDatei;
delete from soundDatei;
delete from soundDatei where sid = 1;

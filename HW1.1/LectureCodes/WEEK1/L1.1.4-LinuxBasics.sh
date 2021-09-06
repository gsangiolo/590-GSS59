
#------------------------------------------------------------------------------------
#DESCRIPTION
#------------------------------------------------------------------------------------

#ASSIGNMENT: 
      #0) OPEN THIS FILE AND A COMMAND LINE SIDE BY SIDE
      #1) ONE BY ONE, UNCOMMENT EACH BLOCK OF CODE, 
            #RUN THE SHELL SCRIPT, OBSERVE THE OUTPUT 
      #2) ADD SHORT COMMENTS AS NEEDED, THEN COMMENT THE BLOCK OUT AGAIN
            #GOOGLE THE COMMANDS IF YOU NEED TO 
            #OR USE "man" FOR DETAILS 
            #OR COME SEE ME 
      #3) MOVE ONTO THE NEXT BLOCK  
      #4) IF A COMMAND HAS ALREADY BEEN COMMENTED ON THEN YOU DONT NEED TO COMMENT A SECOND TIME 

      #NOTE: BLOCK COMMENT IN GEDIT IS cntl-m IN SUBL TEXT ITS cntl-/
      #NOTE --> THE CHARACTER "#" COMMENTS OUT A LINE IN A SHELL SCRIPT


#------------------------------------------------------------------------------------


sleep_time=5  #define variable with sleep time in seconds
DATE=$(date -Is) #save date as variable $(date +"%Y-%m-%d")
echo $DATE


# #---------------------------
# #FILESYSTEM: 
# #---------------------------

# echo "------------------------"
# echo "EXPLORE THE LINUX FILE SYSTEM"
# echo "------------------------"
# cd / 
# pwd
# ls 

# echo "A------------------------"; sleep $sleep_time
# cd /home/
# pwd
# ls 

# echo "B------------------------"; sleep $sleep_time
# cd ~/; 
# pwd
# echo "----------"
# ls -ltr 
# echo "----------"
# ls *;
# echo "----------"
# ls -a
# echo "----------"
# du -csh *


# #---------------------------
# #MANUALS: 
# #---------------------------
# man pwd
# man ls
# man echo
# exit



#---------------------------
#BLOCK: 
#---------------------------



echo "A------------------------"; sleep $sleep_time
cd ~/Documents;
ls

echo "B------------------------"; sleep $sleep_time

rm -rf example_directory
mkdir example_directory
ls


echo "WARNING: BE EXTREMELY CAREFUL WITH 'rm -rf'"
echo "      ESPECIALLY COUPLED THE WILDCARD VARIABLE * OR SUDO (SUPERUSER STATUS)"
echo "      FOR EXAMPLE RUNNING  'cd /; sudo rm -rf *'"
echo "      WOULD DELETE THE ENTIRE OPERATING SYSTEM AND EVERY FILE ON THE LINUX MACHINE"


echo "C------------------------"; sleep $sleep_time

cd example_directory
ls 
echo "im writing to a file" > file1.dat
echo "hello computer" > file2.dat
echo "hello human" >> file2.dat
ls 
more file*.dat
rm file1.dat
ls
more file2.dat
> file2.dat
more file2.dat








##EXPLORE LIST OPTIONS 
#ls -a            #SHOW HIDDEN FILES













#echo "
#NOTE: TO USE THIS SCRIPT SIMPLY OPEN THE BOTH THE FILE AND TERMINAL SIDE-BY-SIDE. RUN THE SCRIPT USING './LinuxBasics-1.sh'. IT WILL INCREMENTALLY OUTPUT EXPLAINATIONS. FOLLOW ALONG INSIDE THE SHELL SCRIPT TO WATCH WHAT THE COMMANDS OUTPUT
#"



#echo "-------------------------------"
#echo "PART-1: BASICS"
#echo "-------------------------------"

#echo "0) echo: one of the simpliest commands is 'echo'. It prints whatever is fed to it to the screen"

#sleep 4  #pause script execution for 4 seconds 

#echo "1) Any command that can be excuted from the command line can also be excuted squentially in am executable file known as a shell script, usually wiht extension '.sh'. To make the file executable, you need to change the file permission using the commang 'chmod a+x file_name.sh'"
#sleep 4

#exit

#echo "
#2) we can define variables 'sleep_time=5'
#"
#sleep 4

#sleep_time=6 #define variable with time to sleep in seconds 

#echo "3) variables are referenced using a $ at the beginning, for example sleep_time="$sleep_time

#sleep $sleep_time

#echo "4) In linux, folders are called directories, the entire linux system is stored in a hieracrhical directory tree, to see where you are use pwd which stands for print working directory"

#pwd

#sleep $sleep_time



#exit

##IMPORTANT LOCATIONS 

## /
## IS THE BOTTOME OF THE DIRECTORY TREE, SHOULD ALMOST NEVER EDIT THINGS HERE
## IT IS WHERE THE OPERATING SYSTEM AND 

## RUN THE FOLLOWING COMMANDS
#cd / 
#ls 
#pwd 
#sleep 2 



##GROUPS

##SUPERUSERS --> FULL PERMISSION (CANT DO VERY BAD THINGS)
#     # sudo rm -rf /  WOULD DELETE THE OPERATING SYSTEM AND ALL FILES
##USERS --> LIMITED FILE PERMISSIONS (CANT BREAK THINGS TOO BADLY)


##NAVIGATING THE DIRECTORY TREE


#FOR EXAMPLE, THE BLOCK 

##---------------------------
##BLOCK
##---------------------------

#cd /
#ls  
#pwd
#cd /

#WOULD BECOME THE FOLLOWING

##---------------------------
##BLOCK
##---------------------------

#cd /             #NAVIGATE TO ROOT DIRECTORY / 
#ls         #LIST TO THE SCREEN WHAT IS IN /
#pwd        #PRINT THE ABSOLUTE PATH OF THE CURRENT WORKING DIRECTORY TO THE SCREEN 
#cd /       
# GIT Setup

git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin <https://github.com/nand0l/react-examapp.git>
git remote add secondary-remote <https://git-codecommit.eu-west-1.amazonaws.com/v1/repos/react-examapp>
git push -u origin main
git push -u secondary-remote main

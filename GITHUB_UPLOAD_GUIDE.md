# GitHub에 올리는 방법 (완벽 가이드)

## 📌 한 문장 요약

**"모든 파일을 같은 폴더에 넣고, 깃허브에 업로드하면, 다른 사람들도 `streamlit run app.py`만으로 사용할 수 있습니다"**

---

## 0️⃣ 사전 준비 (1회만)

### Step 1: GitHub 계정 만들기
```
https://github.com/signup
```

### Step 2: Git 설치
```bash
# Windows
https://git-scm.com/download/win
# 설치 후 재부팅

# Mac
brew install git

# Linux
sudo apt-get install git

# 설치 확인
git --version
```

### Step 3: Git 설정
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 확인
git config --global --list
```

---

## 1️⃣ GitHub에 새 저장소 만들기

### 웹에서 하는 방법 (가장 간단)

**Step A**: https://github.com/new 접속

**Step B**: 저장소 정보 입력
```
Repository name: DRT-Tool
  └─ 또는 다른 이름 (중복 없어야 함)

Description: 
  └─ DRT (Distribution of Relaxation Times) analysis web tool
  
Public 또는 Private 선택
  └─ Public: 누구나 볼 수 있음 (추천)
  └─ Private: 자신만 볼 수 있음

✅ Add a README file (체크 안 함, 우리가 만들었으니까)
✅ Add .gitignore (체크 안 함, 우리가 만들었으니까)

"Create repository" 클릭
```

**결과**: `https://github.com/YOUR_USERNAME/DRT-Tool` 생성됨

---

## 2️⃣ 로컬 폴더 준비

### Step A: 작업 폴더로 이동

```bash
# 터미널/명령프롬프트에서:

# Windows
cd C:\Users\YourName\Documents\DRT-Tool

# Mac/Linux
cd ~/Documents/DRT-Tool
```

### Step B: 파일 확인

```bash
ls -la   # Mac/Linux
dir      # Windows

# 다음 파일이 모두 있나?
# ✅ app.py
# ✅ drt_core.py
# ✅ requirements.txt
# ✅ README.md
# ✅ .gitignore
# ✅ LICENSE (선택)
```

---

## 3️⃣ 로컬에서 Git 초기화

### Step A: Git 저장소 초기화

```bash
cd 프로젝트_폴더

# Git 초기화
git init

# 확인
git status
# 출력:
# On branch master
# 
# No commits yet
# 
# Untracked files:
#   (use "git add <file>..." to include in what will be committed)
#         .gitignore
#         README.md
#         app.py
#         drt_core.py
#         requirements.txt
```

### Step B: 모든 파일 추가

```bash
git add .

# 확인
git status
# 출력:
# On branch master
# 
# Initial commit
# 
# Changes to be committed:
#   (use "rm --cached <file>..." to unstage)
#         new file:   .gitignore
#         new file:   README.md
#         new file:   app.py
#         new file:   drt_core.py
#         new file:   requirements.txt
```

### Step C: 첫 커밋

```bash
git commit -m "Initial commit: DRT analysis tool"

# 출력:
# [master (root-commit) abc1234] Initial commit: DRT analysis tool
#  5 files changed, 1000 insertions(+)
#  create mode 100644 .gitignore
#  create mode 100644 README.md
#  create mode 100644 app.py
#  create mode 100644 drt_core.py
#  create mode 100644 requirements.txt
```

---

## 4️⃣ GitHub에 업로드

### Step A: 원격 저장소 연결

```bash
# GitHub에서 복사한 URL 사용:
# https://github.com/YOUR_USERNAME/DRT-Tool.git

git remote add origin https://github.com/YOUR_USERNAME/DRT-Tool.git

# 확인
git remote -v
# 출력:
# origin  https://github.com/YOUR_USERNAME/DRT-Tool.git (fetch)
# origin  https://github.com/YOUR_USERNAME/DRT-Tool.git (push)
```

### Step B: 업로드 (Push)

```bash
git push -u origin master

# 또는 (최신 Git)
git push -u origin main
```

**처음 실행하면 로그인 창 나타남**
- GitHub 아이디 입력
- 비밀번호 입력 (또는 Personal Access Token)

**완료!** 🎉

---

## 5️⃣ GitHub 페이지에서 확인

### Step A: 웹 확인

```
https://github.com/YOUR_USERNAME/DRT-Tool
```

들어가면 다음이 보여야 함:
```
DRT-Tool
📝 DRT (Distribution of Relaxation Times) analysis web tool

📂 Files:
  - .gitignore
  - README.md
  - app.py
  - drt_core.py
  - requirements.txt

# 분석

DRT (Distribution of Relaxation Times) 분석을 위한 웹 기반 도구입니다...
(README.md의 내용이 자동으로 표시됨)
```

### Step B: Clone 테스트

다른 컴퓨터에서:
```bash
git clone https://github.com/YOUR_USERNAME/DRT-Tool.git
cd DRT-Tool

pip install -r requirements.txt
streamlit run app.py
```

✅ 작동하면 성공!

---

## 🔄 이후 업데이트하기

### 코드 수정 후 업로드

```bash
# 1. 파일 수정 (예: app.py 개선)

# 2. 변경사항 확인
git status

# 3. 변경파일 추가
git add .

# 또는 특정 파일만:
git add app.py

# 4. 커밋 (메시지 필수)
git commit -m "Improve UI layout and performance"

# 5. 업로드
git push

# 끝!
```

---

## 📋 주요 Git 명령어

```bash
# 저장소 상태 확인
git status

# 변경사항 보기
git diff

# 커밋 히스토리 보기
git log

# 마지막 커밋 수정
git commit --amend

# 변경 취소 (주의!)
git checkout -- filename

# 이전 버전으로 돌아가기
git revert HEAD
```

---

## ❌ 실수했을 때

### 실수 1: "파일을 잘못 업로드했는데?"

**해결**:
```bash
# GitHub에서 파일 삭제 (웹에서 클릭)
# 또는 로컬에서:
git rm filename
git commit -m "Remove unwanted file"
git push
```

### 실수 2: "비밀번호/API 키를 업로드했는데?"

**긴급 조치**:
```bash
# 파일 히스토리에서도 제거
git filter-branch --tree-filter 'rm -f filename' HEAD

# GitHub도 이력에서 제거
git push --force
```

⚠️ **중요**: 실제로는 더 복잡합니다. GitHub Secret Scanning으로 자동 감지됩니다.

### 실수 3: ".gitignore를 나중에 추가했는데?"

```bash
# 캐시 제거
git rm -r --cached .

# 다시 추가
git add .

# 커밋
git commit -m "Update .gitignore"

# 푸시
git push
```

---

## 🌟 좋은 연습

### 최고의 커밋 메시지 작성

✅ **좋은 예**:
```
"Fix bug in peak detection algorithm"
"Add automatic lambda selection (GCV)"
"Improve documentation and README"
"Refactor drt_core.py for performance"
```

❌ **안 좋은 예**:
```
"update"
"fix"
"asdf"
"수정됨"
```

### 정기적으로 업데이트

```bash
# 매주 1회 정도
git commit -m "Weekly update"
git push
```

---

## 📈 GitHub에서 공개하기

### Step 1: README가 잘 쓰여졌는지 확인

- ✅ 설명
- ✅ 설치 방법
- ✅ 사용 방법
- ✅ 예제
- ✅ 라이센스

### Step 2: Topics 추가

GitHub 페이지 우측 상단:
```
Add topics:
  - drt
  - electrochemistry
  - eis
  - impedance-spectroscopy
  - python
  - streamlit
```

### Step 3: License 추가

```
Add license → MIT License
```

### Step 4: 다른 프로젝트에 링크

```
About 섹션:
- Description 작성
- Website 입력 (선택)
- Sponsored link (선택)
```

---

## 🚀 공유하기

### 완료 후 공유 방법

```
# 1. 친구에게 공유
친구: "클론하고 싶어"
당신: "git clone https://github.com/YOUR_USERNAME/DRT-Tool.git"

# 2. 논문에 링크
"Our DRT analysis tool is available at:
 https://github.com/YOUR_USERNAME/DRT-Tool"

# 3. SNS/블로그
"새로운 DRT 분석 도구를 만들었습니다! 
 GitHub: https://github.com/YOUR_USERNAME/DRT-Tool"

# 4. 학회/세미나 발표
"https://github.com/YOUR_USERNAME/DRT-Tool
 에서 코드를 다운로드할 수 있습니다"
```

---

## 📊 추적 통계 보기

GitHub 페이지에서:
- **Insights** → **Traffic** → 방문자 수 확인
- **Stargazers** → 즐겨찾기 수
- **Forks** → 복제 수
- **Issues** → 사용자 피드백

---

## 💡 팁

### Tip 1: 좋은 README 만들기
- 이미지/GIF 포함
- 설치 단계 명확
- 실행 예제 포함
- 라이센스 명시

### Tip 2: 정기적 업데이트
- 버그 수정
- 기능 추가
- 문서 개선

### Tip 3: 시작하기 배지 추가
README.md에 추가:
```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/YOUR_USERNAME/DRT-Tool/app.py)

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/DRT-Tool?style=social)](https://github.com/YOUR_USERNAME/DRT-Tool)
```

---

## 🎯 전체 흐름 요약

```
1. GitHub 계정 만들기
   ↓
2. 새 저장소 만들기 (DRT-Tool)
   ↓
3. 로컬 폴더에 모든 파일 준비
   ├─ app.py
   ├─ drt_core.py
   ├─ requirements.txt
   ├─ README.md
   ├─ .gitignore
   └─ LICENSE
   ↓
4. git init (저장소 초기화)
   ↓
5. git add . (파일 추가)
   ↓
6. git commit -m "Initial commit" (커밋)
   ↓
7. git remote add origin https://... (원격 저장소 연결)
   ↓
8. git push -u origin master (업로드)
   ↓
9. GitHub 페이지 확인
   ↓
10. 완료! 🎉
```

---

## ✅ 체크리스트

- [ ] GitHub 계정 생성
- [ ] Git 설치 및 설정
- [ ] 새 저장소 생성 (온라인)
- [ ] 로컬 폴더에 모든 파일 배치
- [ ] git init 실행
- [ ] git add . 실행
- [ ] git commit 실행
- [ ] git remote add 실행
- [ ] git push 실행
- [ ] GitHub 페이지에서 파일 확인
- [ ] 다른 컴퓨터에서 clone 테스트
- [ ] streamlit run app.py 작동 확인

---

## 🎉 완료!

이제 당신의 DRT 분석 도구는 전 세계가 접근 가능한 공개 프로젝트입니다!

```
https://github.com/YOUR_USERNAME/DRT-Tool
```

**다른 연구자들이 `git clone`으로 다운로드하고 사용할 수 있습니다.** ✅

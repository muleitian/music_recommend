from hashlib import sha1
from django.http import JsonResponse
from django.shortcuts import render, redirect,HttpResponseRedirect,reverse
from .models import MusicBrowser,UserInfo
from django.core.paginator import Paginator
from . import user_decorator



# Create your views here.
def register(request):
    context={
        'title':'用户注册',
    }
    return render(request, 'df_user/register.html', context)
def register_handle(request):

    username=request.POST.get('user_name')
    password=request.POST.get('pwd')
    confirm_pwd=request.POST.get('confirm_pwd')
    age=request.POST.get("age")
    gender=request.POST.get("gender")
    email=request.POST.get('email')



    if password!=confirm_pwd:
        return redirect('/user/register/')
    s1=sha1()
    s1.update(password.encode('utf8'))
    encrypted_pwd=s1.hexdigest()

    UserInfo.objects.create(uname=username,upwd=encrypted_pwd,uemail=email,uage=int(age),ugender=gender)


    context={
        'title':'用户登录',
        'username':username
    }
    return render(request, 'df_user/login.html', context)
def register_exist(request):
    username=request.GET.get('uname')
    count=UserInfo.objects.filter(uname=username).count()
    return JsonResponse({'count':count})

def login(request):
    uname=request.COOKIES.get('uname','')
    context={
        'title':'用户登录',
        'error_name':0,
        'error_pwd':0,
        'uname':uname,
    }
    return render(request, 'df_user/login.html', context)

def login_handle(request):
    uname=request.POST.get('username')
    upwd=request.POST.get('pwd')
    jizhu=request.POST.get('jizhu',0)
    users=UserInfo.objects.filter(uname=uname)
    if len(users)==1:
        s1=sha1()
        s1.update(upwd.encode('utf8'))
        if s1.hexdigest()==users[0].upwd:
            url=request.COOKIES.get('url','/')
            red=HttpResponseRedirect(url)
            if jizhu!=0:
                red.set_cookie('uname',uname)
            else :
                red.set_cookie('uname','',max_age=-1)
            request.session['user_id']=users[0].id
            request.session['user_name']=uname
            return red
        else:
            context={
                'title':"用户名登陆",
                'error_name':0,
                'error_pwd':1,
                'uname':uname,
                'upwd':upwd
            }
            return render(request,'df_user/login.html',context)
    else:
        context={
            'title':'用户名登陆',
            'error_name':1,
            'error_pwd':0,
            'uname':uname,
            'upwd':upwd,
        }
        return render(request,'df_user/login.html',context)
def logout(request):
    request.session.flush()
    return redirect(reverse("df_goods:index"))

@user_decorator.login
def info(request):
    username=request.session.get('user_name')
    user=UserInfo.objects.filter(uname=username).first()
    browser_musics=MusicBrowser.objects.filter(user=user).order_by("-browser_time")
    music_list=[]
    if browser_musics:
        music_list=[browser_music.music for browser_music in browser_musics ]
        explain='最近浏览'
    else:
        explain='无最近浏览'
    context={
        'title':'用户中心',
        'page_name':1,
        'user_name':username,
        'music_list':music_list,
        'explain':explain,
    }
    return render(request,'df_user/user_center_info.html',context)
from django.shortcuts import render
from django.core.paginator import Paginator
from datetime import datetime
from df_user.models  import MusicBrowser
from django.db.models import Q
import random
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('mysql+pymysql://root:19981002@localhost:3306/musicre?charset=utf8')
df1 = pd.read_sql_table('song_recm', engine)
# Create your views here.
from .models import MusicInfo,TypeInfo
from .Recommenders import *
def index(request):

    typelist=TypeInfo.objects.all()
    type0=typelist[0].musicinfo_set.order_by('-id')[0:4]
    type01=typelist[0].musicinfo_set.order_by('-mclick')[0:4]
    type1=typelist[1].musicinfo_set.order_by('-id')[0:4]
    type11=typelist[1].musicinfo_set.order_by('-mclick')[0:4]
    type2=typelist[2].musicinfo_set.order_by('-id')[0:4]
    type21=typelist[2].musicinfo_set.order_by('-mclick')[0:4]
    type3=typelist[3].musicinfo_set.order_by('-id')[0:4]
    type31=typelist[3].musicinfo_set.order_by('-mclick')[0:4]
    type4=typelist[4].musicinfo_set.order_by('-id')[0:4]
    type41=typelist[4].musicinfo_set.order_by('-mclick')[0:4]
    type5=typelist[5].musicinfo_set.order_by('-id')[0:4]
    type51=typelist[5].musicinfo_set.order_by('-mclick')[0:4]
    if request.session.get('user_id') is not None:
        typelist = TypeInfo.objects.all()
        type0 = []
        type1 = []
        type2 = []
        type3 = []
        type4 = []
        type5 = []
        type01 = typelist[0].musicinfo_set.order_by('-mclick')[0:4]
        type11 = typelist[1].musicinfo_set.order_by('-mclick')[0:4]
        type21 = typelist[2].musicinfo_set.order_by('-mclick')[0:4]
        type31 = typelist[3].musicinfo_set.order_by('-mclick')[0:4]
        type41 = typelist[4].musicinfo_set.order_by('-mclick')[0:4]
        type51 = typelist[5].musicinfo_set.order_by('-mclick')[0:4]

        n1, n2, n3, n4, n5, n6 = recommender_list()
        for i in n1:
            type0.append(MusicInfo.objects.get(id=i))
        for i in n2:
            type1.append(MusicInfo.objects.get(id=i))
        for i in n3:
            type2.append(MusicInfo.objects.get(id=i))
        for i in n4:
            type3.append(MusicInfo.objects.get(id=i))
        for i in n5:
            type4.append(MusicInfo.objects.get(id=i))
        for i in n6:
            type5.append(MusicInfo.objects.get(id=i))

        context = {
            'title': "首页",
            'type0': type0, 'type01': type01,
            'type1': type1, 'type11': type11,
            'type2': type2, 'type21': type21,
            'type3': type3, 'type31': type31,
            'type4': type4, 'type41': type41,
            'type5': type5, 'type51': type51,

        }
        return render(request, 'df_music/index.html', context)

    context = {
        'title': "首页",
        'type0': type0, 'type01': type01,
        'type1': type1, 'type11': type11,
        'type2': type2, 'type21': type21,
        'type3': type3, 'type31': type31,
        'type4': type4, 'type41': type41,
        'type5': type5, 'type51': type51,

        }
    return render(request, 'df_music/index.html', context)








def song_list(request,tid,pindex,sort):
    typeinfo=TypeInfo.objects.get(pk=int(tid))
    news=typeinfo.musicinfo_set.order_by('-id')[0:2]
    songs_list=[]

    if sort=='1':
        songs_list=MusicInfo.objects.filter(mtype=int(tid)).order_by('-mclick')[0:100]
    if sort=="2":
        songs_list=MusicInfo.objects.filter(mtype=int(tid)).order_by('-id')[0:100]

    paginator=Paginator(songs_list,10)
    page=paginator.page(int(pindex))
    context={
        'title':'歌曲列表',
        'page':page,
        'paginator':paginator,
        'typeinfo':typeinfo,
        'sort':sort,
        'news':news,

    }
    return render(request,'df_music/list.html',context)
def detail(request,mid):
    music_id=mid
    music_tit=MusicInfo.objects.get(pk=int(music_id)).mtitle
    r1 = df1.loc[df1['song_title'] == music_tit, ['song_1', 'song_2', 'song_3']]
    r2=r1.to_numpy()[0]
    song_rec=[]
    for t in r2:
        song=MusicInfo.objects.get(mtitle=t)
        song_rec.append(song)
    songs=MusicInfo.objects.get(pk=int(music_id))
    songs.mclick=songs.mclick+1
    songs.save()

    news=songs.mtype.musicinfo_set.order_by('-mclick')[0:4]

    context={
        'title':songs.mtype.title,
        'songs':songs,
        'news':news,
        'id':music_id,
        'song_rec':song_rec,
    }
    response=render(request,'df_music/detail.html',context)




    user_id=request.session["user_id"]
    if user_id is None:
        return response
    try:
        browsed_music=MusicBrowser.objects.get(user_id=int(user_id),music_id=int(music_id))
    except Exception:
        browsed_music=None
    if browsed_music:
        browsed_music.browser_time=datetime.now()
        browsed_music.save()
    else:
        MusicBrowser.objects.create(user_id=int(user_id),music_id=int(music_id))
        browsed_music=MusicBrowser.objects.filter(user_id=int(user_id))
        browsed_music_count=browsed_music.count()
        if browsed_music_count>5:
            ordered_music=browsed_music.order_by("-brower_time")
            for _ in ordered_music[5:]:
                _.delete()
    return response


def ordinary_search(request):
    search_keywords=request.GET.get('q','')
    pindex=request.GET.get('pindex',1)
    search_status=1


    music_list=MusicInfo.objects.filter(
        Q(mtitle__icontains=search_keywords)|Q(mcontext__icontains=search_keywords)
    ).order_by("mclick")
    if music_list.count()==0:
        search_status=0
        music_list=MusicInfo.objects.all().order_by("mclick")[:4]

    pageinator=Paginator(music_list,4)
    page=pageinator.page(int(pindex))


    context={
        'title':'搜索列表',
        'search_status':search_status,
        'page':page,
        'paginator':pageinator,
    }
    return render(request,'df_music/ordinary_search.html',context)


def recommender_list():
    item_similarity_recommender_py()
    popularity_recommender_py()
    SVD()
    user_popularity()


    n=random.sample(range(0,500),24)
    return split_list_n_list(n,6)



def split_list_n_list(origin_list, n):
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1

    for i in range(0, n):
        yield origin_list[i * cnt:(i + 1) * cnt]


















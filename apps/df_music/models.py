from django.db import models
from datetime import datetime
from tinymce.models import HTMLField
# Create your models here.
#音乐分类
class TypeInfo(models.Model):
    isDelete=models.BooleanField(default=False)
    title=models.CharField(max_length=20,verbose_name="分类")

    class Meta:
        verbose_name="音乐类型"
        verbose_name_plural=verbose_name


    def __str__(self):
        return self.title

#具体音乐信息
class MusicInfo(models.Model):
    isDelete=models.BooleanField(default=False)
    mtitle=models.CharField(max_length=40,verbose_name="音乐名称")
    mpic=models.ImageField(verbose_name="音乐图片",upload_to="df_music/image/%Y/%m",null=True,blank=True)
    mclick=models.IntegerField(verbose_name="点击量",default=0,null=False)
    mcontext=HTMLField(max_length=200,verbose_name="详情")
    mtype=models.ForeignKey(TypeInfo,on_delete=models.CASCADE,verbose_name="分类")

    class Meta:
        verbose_name="歌曲"
        verbose_name_plural=verbose_name
    def __str__(self):
        return self.mtitle








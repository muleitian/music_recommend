from django.db import models
from datetime import datetime
from df_music.models import MusicInfo
# Create your models here.
class UserInfo(models.Model):
    uname=models.CharField(max_length=20,verbose_name="用户名",unique=True)
    upwd=models.CharField(max_length=40,verbose_name="用户密码",blank=False)
    uage=models.IntegerField(verbose_name="年龄")
    ugender=models.CharField(max_length=1,verbose_name="性别")
    uemail=models.EmailField(verbose_name="邮箱",unique=True)
    uphone=models.CharField(max_length=11,default="",verbose_name="手机号")

    class Meta:
        verbose_name="用户信息"
        verbose_name_plural=verbose_name
    def __str__(self):
        return self.uname

class MusicBrowser(models.Model):
    user=models.ForeignKey(UserInfo,on_delete=models.CASCADE,verbose_name="用户ID")
    music=models.ForeignKey(MusicInfo,on_delete=models.CASCADE,verbose_name="歌曲ID")
    browser_time=models.DateTimeField(default=datetime.now,verbose_name="浏览时间")

    class Meta:
        verbose_name="用户浏览记录"
        verbose_name_plural=verbose_name
    def __str__(self):
        return "{0}浏览记录{1}".format(self.user.uname,self.music.mtitle)

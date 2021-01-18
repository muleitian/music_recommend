from django.contrib import admin

# Register your models here.
from .models import UserInfo,MusicBrowser
class UserInfoAdmin(admin.ModelAdmin):
    list_display = ["uname","uemail","uphone"]
    list_per_page = 5
    list_filter = ["uname","uphone"]
    search_fields = ["uname","uemail"]
    readonly_fields = ["uname"]

class MusicBrowserAdmin(admin.ModelAdmin):
    list_display = ["user", "music"]
    list_per_page = 50
    list_filter = ["user__uname", "music__mtitle"]
    search_fields = ["user__uname", "music__mtitle"]
    readonly_fields = ["user", "music"]
    refresh_times = [3, 5]
admin.site.site_header="后台管理系统"
admin.site.site_title="后台管理系统"
admin.site.register(UserInfo,UserInfoAdmin)
admin.site.register(MusicBrowser,MusicBrowserAdmin)


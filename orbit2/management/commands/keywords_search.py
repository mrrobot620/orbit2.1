# your_script.py
from django.core.management.base import BaseCommand
from orbit2.models import Pendency
from django.db.models import Q
import django





class Command(BaseCommand):
    help = 'Your script description here'

    def handle(self, *args, **options):
        keyword_query()

def keyword_query():
    keywords_to_check = ['white tshirt']
    
    # Create a Q object for case-insensitive search on each keyword
    q_objects = [Q(keywords__icontains=keyword) for keyword in keywords_to_check]
    
    # Combine Q objects with OR operator
    query = Q()
    for q_obj in q_objects:
        query |= q_obj

    queryset = Pendency.objects.filter(query)

    if queryset.exists():
        # Print only the 'tid' attribute for each matching object
        for pendency_object in queryset:
            print(pendency_object.tid)
    else:
        print("No Pendency objects found with the specified keywords.")

# SAM-Tracker Dataloop Service

This is a Dataloop service for a Segment-Anything Model (SAM) based object tracker.

To run on your own, clone this project and run `pip install -r requirements.txt`

There are several supported SAMs, and the weight files are too big for git. in the `models`
directory you can find the mobile-sam model and a README with download links to other supported models.

in `test_tracker` file you can run the tracker itself locally using `test_local`, or test the service using `test_dl`.
make sure to replace the item id to your item!

### Adding a new SAM
To add a new SAM, simply add a new class in the SAM folder that implements the SAM abstract class.
More information about the functions to override are in the SAM class.
